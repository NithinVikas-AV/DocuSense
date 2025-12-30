from flask import Flask, request, jsonify, send_file, render_template
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
import pdfkit
from serpapi import GoogleSearch

app = Flask(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
project_id = os.getenv("PROJECT_ID")
collection_name = os.getenv("collection_name")
session_id = os.getenv("SESSION_ID")
serpapi_key = os.getenv("SERPAPI_KEY")

# Initialize directories and embeddings
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
uploads_dir = os.path.join(current_dir, "uploads")
idx_name = "FAISS_metadata"

if not os.path.exists(db_dir):
    os.makedirs(db_dir)
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Ensure FAISS database exists
def ensure_faiss_db_exists():
    index_path = os.path.join(db_dir, f"{idx_name}.faiss")
    if not os.path.exists(index_path):
        dummy_doc = Document(page_content="", metadata={"source": "init"})
        faissdb = FAISS.from_documents([dummy_doc], embeddings)
        faissdb.save_local(folder_path=db_dir, index_name=idx_name)

# File extraction functions
def text_file_extract(file_name):
    file_path = os.path.join(current_dir, "uploads", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    loader = TextLoader(file_path, encoding="utf-8")
    data = loader.load()
    
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

# Initialize LLM and Firestore
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', google_api_key=api_key)
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

# Initialize FAISS
ensure_faiss_db_exists()
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

# Create RAG chain
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
        " ** You can refer to the chat history if the retrieved context is not related to the question **"
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

config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

def export_chat(_, **kwargs):
    chat_text = "\n".join(
        f"{'You' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in memory.chat_memory.messages
    )
    export_path = os.path.join(current_dir, "chat_history.pdf")
    # Convert newlines to <br> for better formatting in PDF
    html_content = chat_text.replace("\n", "<br>")
    pdfkit.from_string(html_content, export_path, configuration=config)
    # Return a URL the frontend can use to download
    return f"Chat exported successfully! Download it by clicking the download chats button."

def web_search(query, **kwargs):
    """Uses SerpAPI to search Google and returns a summarized result."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_key,
        "num": 3,
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    if "error" in results:
        return f"SerpAPI error: {results['error']}"
    
    organic_results = results.get("organic_results", [])
    if not organic_results:
        return "No relevant search results found."
    
    combined_text = ""
    for res in organic_results:
        title = res.get("title", "")
        snippet = res.get("snippet", "")
        link = res.get("link", "")
        combined_text += f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n\n"
    
    summary = llm.invoke([
        SystemMessage(content="Summarize the following web search results briefly:"),
        HumanMessage(content=combined_text)
    ]).content
    
    return summary

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
        description="Exports the current chat history to a downloadable PDF file."
    ),
    Tool(
        name="Search Web",
        func=web_search,
        description="Use this to search the web for recent or unknown information and return a brief summary."
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
    # memory=memory,
    handle_parsing_errors=True,
)

# API Endpoints
@app.route('/')
def serve_frontend():
    return render_template('text_chat.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    query = data.get('input')
    if not query:
        return jsonify({'error': 'No input provided'}), 400
    
    try:
        memory.chat_memory.add_message(HumanMessage(content=query))
        response = agent_executor.invoke({"input": query})
        output = response["output"]
        memory.chat_memory.add_message(AIMessage(content=output))
        
        return jsonify({'output': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and (file.filename.endswith('.txt') or file.filename.endswith('.pdf')):
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        try:
            if file.filename.endswith('.txt'):
                text_file_extract(file.filename)
            else:
                pdf_file_extract(file.filename)
            return jsonify({'message': 'File uploaded and processed successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only .txt and .pdf are allowed'}), 400

@app.route('/voice')
def serve_voice_chat():
    return render_template('voice_chat.html')

@app.route('/download/chat_history.pdf')
def download_chat_history():
    file_path = os.path.join(current_dir, 'chat_history.pdf')
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Saving chat history to Firestore...")

        firestore_history.clear()

        for msg in memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                firestore_history.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                firestore_history.add_ai_message(msg.content)
            elif isinstance(msg, SystemMessage):
                # Optional: save system messages too
                firestore_history.add_ai_message(msg.content)

        print("[INFO] Chat history saved successfully. Goodbye!")