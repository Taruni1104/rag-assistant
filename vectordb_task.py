from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load document
loader = TextLoader("my_env/new-Policies.txt")  # Adjust path if needed
documents = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 3. Load embedding model (no API needed)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create Chroma DB
vectordb = Chroma.from_documents(chunks, embedding_model)

# 5. Similarity search
query = "Smoking policy"
results = vectordb.similarity_search(query, k=5)

# 6. Print top 5 matches
print("Top 5 similar chunks for query:", query)
for i, res in enumerate(results, 1):
    print(f"\nResult {i}:\n{res.page_content}")
