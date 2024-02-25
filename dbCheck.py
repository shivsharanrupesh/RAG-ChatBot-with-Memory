import qdrant_client
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Qdrant

QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# create persistant db
client = qdrant_client.QdrantClient(
    url = QDRANT_HOST,
    api_key = QDRANT_API_KEY,
)

#create collection
collection_name = "chat-rag-memory"
vector_config = qdrant_client.http.models.VectorParams(
    size = 1024,
    distance = qdrant_client.http.models.Distance.COSINE
)
client.recreate_collection(
    collection_name = collection_name,
    vectors_config = vector_config,
)

web_links = ["https://hotels-ten.vercel.app/api/hotels"] 
loader = WebBaseLoader(web_links)
document=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(document)

embeddings = CohereEmbeddings(model = "embed-english-v3.0")
print(" embedding docs !")

vector_store = Qdrant(
    client=client,
    collection_name = collection_name,
    embeddings=embeddings
)
vector_store.add_documents(texts)
retriever=vector_store.as_retriever()

