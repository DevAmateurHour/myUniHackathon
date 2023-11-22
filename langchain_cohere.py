#fixing sqlite3 only required on Streamlit Cloud App - likely due to Linux. On Windows - this appears to cause an exception - therefore using Try/Except
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    donothing=None # It's OK if the above causes an exception. If exception, then likely means sqlite3 version is OK on this machine

#imports
from langchain.chat_models import ChatCohere
import cohere  
from langchain.llms import Cohere
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader

def setup_chain(api_key, chunk_size, chunk_overlap, max_tokens, temperature, search_k, verbose):
    #Initalize Cohere
    co = cohere.Client(api_key)
    #I put the text file into u_data folder, you can usde your path
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader("u_data/", silent_errors=False, show_progress=True, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    len(docs)
    embeddings = CohereEmbeddings(cohere_api_key=api_key)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents, embeddings) #db
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    model =ChatCohere(model="command-nightly",max_tokens=max_tokens,temperature=temperature, cohere_api_key=api_key)

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

    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=index.vectorstore.as_retriever(search_kwargs={"k": search_k}), verbose = verbose, chain_type_kwargs={
            "verbose": verbose,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        })
    
    return chain

