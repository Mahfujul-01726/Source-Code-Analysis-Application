from flask import Flask, render_template, jsonify, request
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
import os
import subprocess

from src.helper import load_embedding, repo_ingestion

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load OpenAI embeddings
embeddings = load_embedding()

# VectorDB directory
persist_directory = "db"

# Load vector database
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Initialize LLM and memory
llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
    memory=memory
)

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/chatbot', methods=["POST"])
def gitRepo():
    try:
        user_input = request.form['question']
        repo_urls = [url.strip() for url in user_input.split(',') if url.strip()]

        if not repo_urls:
            return jsonify({"error": "No repository URL provided."}), 400

        # Clone and get paths
        cloned_paths = repo_ingestion(repo_urls)

        # Index each repo
        for path in cloned_paths:
            subprocess.run(["python", "store_index.py", path], check=True)

        # 🔁 Reload vectorstore after indexing new repos
        global vectordb, qa
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
            memory=memory
        )

        return jsonify({"response": f"✅ Successfully cloned and indexed: {repo_urls}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg.strip()
    print(f"💬 User: {input_text}")

    if input_text.lower() == "clear":
        # Clear repo directory
        print("🧹 Clearing 'repo' directory")
        if os.name == 'nt':
            os.system("rmdir /s /q repo")  # Windows
        else:
            os.system("rm -rf repo")      # Linux/Mac
        return "🧹 Repo directory cleared."

    try:
        result = qa(input_text)
        print(f"🤖 Answer: {result['answer']}")
        return str(result["answer"])
    except Exception as e:
        print(f"❌ Error: {e}")
        return "❌ Sorry, an error occurred."


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
