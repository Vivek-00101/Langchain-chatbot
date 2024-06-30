import os
from flask import Flask, request, jsonify
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone

# Function to extract data from a URL using Langchain's WebBaseLoader
def extract_data(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    return document_chunks

# Function to create embeddings and store them in Pinecone vector store
def create_embeddings_and_store(data):
    embeddings = OpenAIEmbeddings(api_key='openai_API_key')
    
    # Initialize Pinecone with your API key
    pinecone_api_key = os.getenv("API_Key")
    if pinecone_api_key is None:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")

    pinecone_env = os.getenv("us-east-1")
    
    # Create a Pinecone instance
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    
    index_name = 'langchain-demo'

    # Check if index exists, otherwise create it
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=512,
            metric='cosine'  # You can choose a suitable metric
        )

    index = pc.index(index_name)
    
    embeddings_list = []
    for i, chunk in enumerate(data):
        embedding = embeddings.generate_embedding(chunk)
        embeddings_list.append((str(i), embedding, {}))
    
    index.upsert(vectors=embeddings_list)

# Set up Flask application
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json['query']
    # Implement chatbot logic using Langchain and Pinecone here

    return jsonify({"response": "This is a placeholder response."})

if __name__ == '__main__':
    # Replace with actual URL from where you want to extract data
    url = "https://brainlox.com/courses/category/technical"
    data = extract_data(url)
    create_embeddings_and_store(data)
    app.run(debug=True)
