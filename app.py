import streamlit as st
import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="UChicago Data Science Assistant",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 UChicago MS in Applied Data Science Assistant")
st.markdown("Ask me anything about the MS in Applied Data Science program!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for RAG chain
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.vectorstore = None

os.environ['OPENAI_API_KEY'] = os.getenv("UChiRAG_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

urls_df = pd.read_csv("urls.csv")
urls = list(urls_df['url'])
urls = [url[:-1] if url.endswith("/") else url for url in urls]

# Load documents
loader = WebBaseLoader(web_paths=urls)
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
splits = text_splitter.split_documents(docs)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 6}
)

# Define prompt
prompt_template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    Do not create new information that is not in the context. Only provide information that exists in the context you are given.
    However, if you are given multiple contexts sourcing from different links, differentiate them in your answer.
    Always provide the URLs to the user to the pages where they can get the information.
    Be aware that there is a joint MBA/MS program that has different requirements/curriculum.
    There is also an online program for the MS that is slightly different than the in-person (default) MS program.
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
prompt = ChatPromptTemplate.from_template(prompt_template)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Format docs function
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
                
# Store in session state
st.session_state.rag_chain = rag_chain
st.session_state.vectorstore = vectorstore

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the MS in Applied Data Science program..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with LangChain, OpenAI, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)