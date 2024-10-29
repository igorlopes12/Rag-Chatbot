'''This script implements a Streamlit-based application for interacting with documents using a chatbot interface.
The application allows users to upload PDF files, processes the content into chunks, and stores them in a vector store.
Users can then ask questions, and the application retrieves relevant context from the vector store to generate answers using a language model.
Functions:
    process_pdf(file):
    load_existing_vector_store():
            Chroma or None: The initialized vector store if the directory exists, otherwise `None`.
    add_to_vector_store(chunks, vector_store=None):
    ask_question(model, query, vector_store):
Global Variables:
    persist_directory (str): The directory where the vector store is persisted.
    vector_store (Chroma or None): The loaded or newly created vector store.
Streamlit Components:
    st.set_page_config: Configures the Streamlit page with a title and icon.
    st.header: Displays the main header of the application.
    st.sidebar: Displays the sidebar for uploading documents and selecting the model.
    st.file_uploader: Allows users to upload PDF files.
    st.spinner: Displays a spinner while files are being uploaded.
    st.selectbox: Allows users to select a language model from a dropdown.
    st.chat_input: Displays an input box for users to ask questions.
    st.chat_message: Displays chat messages in the chat interface.
    '''
import os
import tempfile

import streamlit as st

from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'


def process_pdf(file):
    """
    Processes a PDF file and splits its content into chunks.
    Args:
        file (file-like object): The PDF file to be processed.
    Returns:
        list: A list of document chunks, where each chunk is a portion of the PDF content.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 400,
    )
    chunks = text_spliter.split_documents(documents=docs)
    return chunks

def load_existing_vector_store():
    """
    Loads an existing vector store from the specified directory if it exists.

    This function checks if the directory specified by `persist_directory` exists.
    If it does, it initializes and returns a `Chroma` vector store using the 
    `OpenAIEmbeddings` embedding function. If the directory does not exist, 
    the function returns `None`.

    Returns:
        Chroma or None: The initialized vector store if the directory exists, 
        otherwise `None`.
    """
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
            )
        return vector_store
    return None

def add_to_vector_store(chunks, vector_store=None):
    """
    Adds document chunks to a vector store. If a vector store is provided, 
    the chunks are added to it. Otherwise, a new vector store is created 
    using the provided chunks.

    Args:
        chunks (list): A list of document chunks to be added to the vector store.
        vector_store (optional): An existing vector store to which the chunks 
                                 will be added. If not provided, a new vector 
                                 store will be created.

    Returns:
        The updated or newly created vector store.
    """
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents = chunks,
            embedding = OpenAIEmbeddings(),
            persist_directory = persist_directory,
        )
    return vector_store

def ask_question(model, query, vector_store):
    """
    Asks a question using a language model and a vector store for context retrieval.
    Args:
        model (str): The name or identifier of the language model to use.
        query (str): The question to be asked.
        vector_store (VectorStore): The vector store used to retrieve relevant context.
    Returns:
        str: The answer to the question, formatted in markdown with visualizations if applicable.
    """
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever()

    system_prompt = '''

    Use the context to answer the questions.  
    If no answer is found in the context, explain that no information is available.  
    Respond in markdown format and with visualizations.
    Context: {context}
'''
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)
    question_answer_chain = create_stuff_documents_chain(
        llm = llm,
        prompt = prompt,
    )

    chain = create_retrieval_chain(
        retriever = retriever,
        combine_docs_chain = question_answer_chain,
    )
    chain_response = chain.invoke({'input': query})
    return chain_response.get('answer')


vector_store = load_existing_vector_store()


# Page Config
st.set_page_config(
    page_title="Igor bot",
    page_icon="🤖",
)
st.header("🤖 Chat with your documents (RAG)")


# Sidebar
with st.sidebar:
    st.header('Upload documents')
    uploaded_files = st.file_uploader(
    label="Upload your PDF File",
    type=['pdf'],
    accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner('Uploading...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]

    selected_model = st.sidebar.selectbox(
            label='Select model',
            options=model_options,
    )

# Chat History
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input("How can I help you?")

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    with st.spinner('Searching for an answer...'):
        response = ask_question(
            model = selected_model,
            query = question,
            vector_store = vector_store,
    )

    st.chat_message('ai').write(response)
    st.session_state.messages.append({'role': 'ai', 'content': response})

