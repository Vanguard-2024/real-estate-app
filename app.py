from langchain_groq import ChatGroq
import streamlit as st
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests
import datetime

# Initialize the LLM (ChatGroq)
llm = ChatGroq(api_key="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", model="llama3-70b-8192")

# Initialize the embedding model for document and query embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure that the ChromaDB client is initialized only once
if 'chroma_client' not in st.session_state:
    # Use in-memory client instead of persistent file storage
    st.session_state.chroma_client = Client(memory=True)  # Using in-memory client

# Get Chroma client from session state
client = st.session_state.chroma_client

# Define a new collection name with a timestamp to make it unique
new_collection_name = f"real_estate_docs_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

# Create a new collection with the unique name
collection = client.create_collection(
    name=new_collection_name,
    embedding_function=SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')
)
print(f"New collection '{new_collection_name}' has been created.")

# Function to perform Google search using the Serper API and gather real estate data
def fetch_real_estate_info():
    query = "real estate news market trends investment upto Today"
    api_key = "b4360a6824b8e1af6c15c69528fdf8269808e892"  # Replace with your Serper API key
    
    # Call the Serper API for a Google search
    url = f"https://api.serper.dev/search?api_key={api_key}&q={query}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        snippets = [result.get('snippet', '') for result in data.get("organic", [])]
        document = "\n\n".join(snippets)
        return document
    else:
        return "Sorry, I couldn't retrieve real estate information."

# Function to store real estate information in ChromaDB
def store_real_estate_info_in_db():
    real_estate_document = fetch_real_estate_info()

    if real_estate_document:
        document_id = f"real_estate_info_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        collection.add(
            documents=[real_estate_document],
            metadatas=[{'source': 'real_estate_search', 'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}],
            ids=[document_id]
        )
        print("Real estate information successfully stored in ChromaDB.")
    else:
        print("No real estate information retrieved.")

# Initialize Streamlit app layout
st.title('Ask Real360')

# Initialize session state for chat messages if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Define domain context
domain_context = """
You are a real estate AI assistant. Your knowledge covers the real estate market, property values, investment strategies, home buying and selling processes, mortgages, and other related real estate topics. Please provide informative, helpful, and clear responses based only on real estate topics. Do not provide information outside the real estate domain.
"""

# Function to query the vector database for relevant documents
def query_vector_db(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results['documents']  # Return all top n results as context

# Input for new user prompt
prompt = st.chat_input('Pass Your Prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    relevant_docs = query_vector_db(prompt)
    relevant_docs_str = [str(doc) for doc in relevant_docs] 
    full_prompt = domain_context + " Here are some relevant documents:\n" + "\n".join(relevant_docs_str) + "\n" + prompt
    
    response = llm.invoke([HumanMessage(content=full_prompt)], max_tokens=400, temperature=0.7)

    st.chat_message('assistant').markdown(response.content)
    st.session_state.messages.append({'role': 'assistant', 'content': response.content})

store_real_estate_info_in_db()
