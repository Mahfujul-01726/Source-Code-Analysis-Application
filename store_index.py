import os
import sys
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from src.helper import load_repo, text_splitter, load_embedding

# Load API key
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Get repo path from command-line argument
if len(sys.argv) < 2:
    print("❌ Please provide the path to the cloned repo as an argument.")
    sys.exit(1)

repo_path = sys.argv[1]

# ✅ Load, split, embed, and store
documents = load_repo(repo_path)
text_chunks = text_splitter(documents)
embeddings = load_embedding()

# ✅ Store vectors
vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory="db")
vectordb.persist()

print(f"✅ Indexing complete for: {repo_path}")
