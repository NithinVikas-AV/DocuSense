import os
from dotenv import load_dotenv
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
import speech_recognition as sr
import pyttsx3

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
project_id = os.getenv("PROJECT_ID")
collection_name = os.getenv("collection_name")
session_id = os.getenv("SESSION_ID")

def ensure_faiss_db_exists():

    index_path = os.path.join(db_dir, f"{idx_name}.faiss")
    if not os.path.exists(index_path):

        dummy_doc = Document(page_content="", metadata={"source": "init"})
        faissdb = FAISS.from_documents([dummy_doc], embeddings)

        faissdb.save_local(folder_path=db_dir, index_name=idx_name)

def text_file_extract(file_name):
    
    file_path = os.path.join(current_dir, "uploads", file_name)

    loader = TextLoader(file_path, encoding="utf-8")
    data = loader.load()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = text_splitter.split_documents(data)

    for chunk in chunks:
        chunk.metadata = {'source': file_name}

    faissdb = FAISS.load_local(
        folder_path=db_dir,
        embeddings=embeddings,
        index_name=idx_name,
        allow_dangerous_deserialization=True,
    )

    faissdb.add_documents(chunks)

    faissdb.save_local(folder_path=db_dir, index_name=idx_name)

def pdf_file_extract(file_name):

    file_path = os.path.join(current_dir, "uploads", file_name)

    loader = PyMuPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
        length_function=len,
    )

    chunks = text_splitter.split_documents(data)

    for chunk in chunks:
        chunk.metadata = {'source': file_name}

    faissdb = FAISS.load_local(
        folder_path=db_dir,
        embeddings=embeddings,
        index_name=idx_name,
        allow_dangerous_deserialization=True,
    )

    faissdb.add_documents(chunks)

    faissdb.save_local(folder_path=db_dir, index_name=idx_name)

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=api_key)

client = firestore.Client(project=project_id)

firestore_history = FirestoreChatMessageHistory(
    session_id=session_id,
    collection=collection_name,
    client=client,
)

initial_system_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.add_message(SystemMessage(content=initial_system_message))

for msg in firestore_history.messages:
    if not isinstance(msg, SystemMessage):
        memory.chat_memory.add_message(msg)

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
idx_name = "FAISS_metadata"

modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs
)

faissdb = FAISS.load_local(
    folder_path=db_dir,
    embeddings=embeddings,
    index_name=idx_name,
    allow_dangerous_deserialization=True,
)

rag_retriever = faissdb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'k': 2, 'score_threshold': 0.2},
)

def create_rag_chain():
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, rag_retriever, contextualize_q_prompt,
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question."
        " ** You can Refer the chat history if the retrieved context is not related to the question. **"
        " If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise. Also Mention the source of the context if you took from the relevant data."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def export_chat(_, **kwargs):
    chat_text = "\n".join(
        f"{'You' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in memory.chat_memory.messages
    )
    with open("chat_history.txt", "w", encoding="utf-8") as f:
        f.write(chat_text)
    return "Conversation exported to chat_history.txt."

rag_chain = create_rag_chain()

tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": memory.chat_memory.messages}
        ),
        description="useful for when you need to answer questions about the context",
    ),
    Tool(
        name="Summarize Conversation",
        func=lambda input, **kwargs: llm.invoke([
            SystemMessage(content="Summarize the following chat history briefly:"),
            *memory.chat_memory.messages
        ]).content,
        description="Summarizes the current conversation so the user can get a quick recap."
    ),
    Tool(
        name="Export Chat",
        func=export_chat,
        description="Exports the current chat to a local text file."
    )

]

react_docstore_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

def handle_user_text_interaction():

    try:
        while True:
            query = input("You: ")
            if query.lower() == "terminate":
                break

            memory.chat_memory.add_message(HumanMessage(content=query))

            response = agent_executor.invoke({"input": query})
            print("Bot:", response["output"])

            memory.chat_memory.add_message(AIMessage(content=response["output"]))
    
    except KeyboardInterrupt:
        print("\n[INFO] Exiting gracefully. Goodbye!")

    firestore_history.clear() 
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            firestore_history.add_user_message(msg.content)
        elif isinstance(msg, AIMessage):
            firestore_history.add_ai_message(msg.content)

    print("Chat history successfully synced to Firestore.")

def handle_user_voice_interaction():
    
    try:
        while True:
            recognizer = sr.Recognizer()

            with sr.Microphone() as source:
                print("Listening... Say something:")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                print("Recognizing...")
                query = recognizer.recognize_google(audio)

                if query == "terminate":
                    break
                
                print(f"You said: {query}")
                        
            except sr.UnknownValueError:
                print("Sorry, could not understand audio.")
            except sr.RequestError as e:
                print(f"Error accessing Google Web Speech API; {e}")

            memory.chat_memory.add_message(HumanMessage(content=query))
            
            response = agent_executor.invoke({"input": query})
            print("Bot:", response["output"])

            memory.chat_memory.add_message(AIMessage(content=response["output"]))

            engine = pyttsx3.init()
            engine.setProperty('rate', 190)
            engine.setProperty('volume', 5.0)

            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[0].id)

            engine.say(response["output"])
            engine.runAndWait()
            engine.stop()

    except KeyboardInterrupt:
        print("\n[INFO] Exiting gracefully. Goodbye!")

    firestore_history.clear() 
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            firestore_history.add_user_message(msg.content)
        elif isinstance(msg, AIMessage):
            firestore_history.add_ai_message(msg.content)

if __name__ == '__main__':
    ensure_faiss_db_exists()
    handle_user_text_interaction()