#imports - comment these out if running locally - required just for streamlit cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#imports
from langchain.chat_models import ChatCohere
from langchain.schema import HumanMessage
import cohere  
from langchain.llms import Cohere
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain #
from langchain.vectorstores import Qdrant #
from langchain.memory import ConversationBufferMemory
import os
from langchain.document_loaders import TextLoader
from chromadb.utils import embedding_functions
import streamlit as st
from dotenv import load_dotenv

if "temperature" not in st.session_state:
    st.session_state.temperature = float(st.secrets["TEMPERATURE"] or 0.75)
    
if "search_k" not in st.session_state:
    st.session_state.search_k = int(st.secrets["SEARCH_K"] or 1)

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = int(st.secrets["CHUNK_SIZE"] or 1000)

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = int(st.secrets["MAX_TOKENS"] or 600)

if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = float(st.secrets["CHUNK_OVERLAP"] or 0.00)

if "chat_input" not in st.session_state:
    st.session_state.chat_input = st.secrets["CHAT_INPUT"] or 'Tell us more about you, your dream and your ambition'

if "init_assistant_message" not in st.session_state:
    st.session_state.init_assistant_message = st.secrets["INIT_ASSISTANCE_MESSAGE"] or "Hey, I'm Ed, your dedicated research assistant! Ready to make your university dreams happen. Where do you want to kick things off?"



@st.cache_resource
def setup_chain():
    #Get Environment Variables
    load_dotenv()
    api_key=st.secrets["COHERE_API_KEY"]
    if api_key == None:
        raise Exception('Require a COHERE_API_KEY to be setup in as an environment variable')
    max_tokens=int(st.secrets["MAX_TOKENS"] or 600)
    
    
    st.title(st.secrets["TITLE"] or 'Welcome to MatchMyUni')
    
    print('max_tokens',st.session_state.max_tokens,'temperature',st.session_state.temperature,'search_k',st.session_state.search_k, 'chunk_size', st.session_state.chunk_size, 'chunk_overlap', st.session_state.chunk_overlap)
    
    #Initalize Cohere
    co = cohere.Client(api_key)
    #I put the text file into u_data folder, you can usde your path
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader("u_data/", silent_errors=False, show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    len(docs)
    embeddings = CohereEmbeddings(cohere_api_key=api_key)
    text_splitter = CharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
    documents = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents, embeddings) #db
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    model =ChatCohere(model="command-nightly",max_tokens=st.session_state.max_tokens,temperature=st.session_state.temperature, cohere_api_key=api_key)

    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=index.vectorstore.as_retriever(search_kwargs={"k": st.session_state.search_k}), verbose = True, chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        })
    
    return chain

chain = setup_chain()

question = ""

#Initialize the streamlit App
def submit_parameters():
    st.session_state.chunk_size = int(st.session_state.chunk_size_parameter)
    st.session_state.search_k = int(st.session_state.search_k_parameter)
    st.session_state.temperature = float(st.session_state.temperature_parameter)
    st.session_state.max_tokens = int(st.session_state.max_tokens_parameter)
    st.session_state.chunk_overlap = float(st.session_state.chunk_overlap_parameter)
    st.session_state.messages = []
    st.cache_resource.clear()

with st.sidebar.form("Parameters"):
    st.slider('Temperature',0.00,1.00,st.session_state.temperature,.05, key="temperature_parameter")
    st.slider('Search K',1,10,st.session_state.search_k,1, key="search_k_parameter")
    st.slider('Chunk Size',200,2500,st.session_state.chunk_size,100, key="chunk_size_parameter")
    st.slider('Max Tokens',100,1000,st.session_state.max_tokens,100, key="max_tokens_parameter")
    st.slider('Chunk Overlap',0.00,0.50,st.session_state.chunk_overlap,0.05, key="chunk_overlap_parameter")
    submitted = st.form_submit_button("Submit and Reset", on_click=submit_parameters)
     
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input(st.session_state.chat_input):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

# First default response from Assistance
response = st.session_state.init_assistant_message

if question != None:
    #response = f"{list(chain(prompt).values())[1]}"
    response = chain.run({"query": question})
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

#query=""
#while query!="stop":
#    query = input("Prompt: ")
#    if query=="stop":
#        print("stoped by user")
#    print(list(chain(query).values())[1])