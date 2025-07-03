import os
from git import Repo
from datetime import datetime
from urllib.parse import urlparse

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


# ✅ Clone GitHub repositories into unique subfolders under 'repo/'
def repo_ingestion(repo_urls):
    cloned_paths = []
    for repo_url in repo_urls:
        try:
            # Extract repo name
            repo_name = os.path.basename(urlparse(repo_url).path).replace(".git", "")
            
            # Create unique folder name using timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_folder_name = f"{repo_name}_{timestamp}"
            repo_path = os.path.join("repo", unique_folder_name)

            print(f"🚀 Cloning {repo_url} into {repo_path}")
            Repo.clone_from(repo_url, to_path=repo_path)

            cloned_paths.append(repo_path)

        except Exception as e:
            print(f"❌ Error cloning {repo_url}: {e}")

    return cloned_paths



# ✅ Load Python source files from the given repo path
def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    return documents



# ✅ Split Python documents into chunks
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = documents_splitter.split_documents(documents)
    return text_chunks



# ✅ Load OpenAI Embedding model
def load_embedding():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings
