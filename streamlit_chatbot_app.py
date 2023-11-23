import streamlit as st
from helper import *
from langchain_cohere import Chain

def initialize():
    chain_parameters, ui_parameters = initialize_parameters()

    #   Get Parameter Variables
    st.session_state.temperature = chain_parameters['temperature']
    st.session_state.search_k = chain_parameters['search_k']
    st.session_state.chunk_size = chain_parameters['chunk_size']
    st.session_state.max_tokens = chain_parameters['max_tokens']
    st.session_state.chunk_overlap = chain_parameters['chunk_overlap']
    st.session_state.verbose = chain_parameters['verbose']
    st.session_state.chat_input = ui_parameters['chat_input']
    st.session_state.init_assistant_message = ui_parameters['init_assistant_message']
    st.session_state.title = ui_parameters['title']

    st.session_state.api_key=st.secrets["COHERE_API_KEY"] if "COHERE_API_KEY" in st.secrets else None
    if st.session_state.api_key == None:
        raise Exception('Require a COHERE_API_KEY to be setup as an Secret') 

    st.session_state.messages = []   
    
    st.session_state.initialized = True

def submit_parameters():
    st.session_state.chunk_size = int(st.session_state.chunk_size_parameter)
    st.session_state.search_k = int(st.session_state.search_k_parameter)
    st.session_state.temperature = float(st.session_state.temperature_parameter)
    st.session_state.max_tokens = int(st.session_state.max_tokens_parameter)
    st.session_state.chunk_overlap = float(st.session_state.chunk_overlap_parameter)
    st.session_state.messages = []
    del st.session_state["chain"]

def streamlit_ui():
    st.title(st.session_state.title)
    
    with st.sidebar.form("Parameters"):
        st.slider('Temperature',0.00,1.00,st.session_state.temperature,.05, key="temperature_parameter")
        st.slider('Search K',1,10,st.session_state.search_k,1, key="search_k_parameter")
        st.slider('Chunk Size',200,2500,st.session_state.chunk_size,100, key="chunk_size_parameter")
        st.slider('Max Tokens',100,1000,st.session_state.max_tokens,100, key="max_tokens_parameter")
        st.slider('Chunk Overlap',0.00,0.50,st.session_state.chunk_overlap,0.05, key="chunk_overlap_parameter")
        submitted = st.form_submit_button("Submit and Reset", on_click=submit_parameters)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    question = ""
    if question := st.chat_input(st.session_state.chat_input):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

    # First default response from Assistance
    response = st.session_state.init_assistant_message

    if question != None:
        response = st.session_state.chain.run(question)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


if "initialized" not in st.session_state:
    initialize()

if "chain" not in st.session_state:
    st.session_state.chain = Chain(st.session_state.api_key, st.session_state.chunk_size, 
                                        st.session_state.chunk_overlap, st.session_state.max_tokens, 
                                        st.session_state.temperature, st.session_state.search_k,
                                        st.session_state.verbose)

streamlit_ui()


