import streamlit as st
import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PlaywrightURLLoader
from dotenv import load_dotenv

load_dotenv()

urls = [
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/career-outcomes",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/instructors-staff",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/our-students",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/events-deadlines",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/how-to-apply",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/capstone-projects",
"https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science",
"https://datascience.uchicago.edu/education/masters-programs/in-person-program",
"https://datascience.uchicago.edu/education/masters-programs/online-program",
"https://datascience.uchicago.edu/education/tuition-fees-aid",
"https://ms-ads.datascience.uchicago.edu",
"https://www.chicagobooth.edu/mba/joint-degree/mba-ms-applied-data-science?sc_lang=en",
"https://datascience.uchicago.edu/education/tuition-fees-aid"]

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

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    langchain_api_key = st.text_input("LangChain API Key (optional)", type="password", value=os.getenv("LANGCHAIN_API_KEY", ""))
    
    # Initialize button
    if st.button("Initialize RAG System", type="primary"):
        if not openai_api_key:
            st.error("Please provide an OpenAI API key!")
        else:
            with st.spinner("Loading documents and building vector store..."):
                try:
                    # Set environment variables
                    os.environ['OPENAI_API_KEY'] = openai_api_key
                    if langchain_api_key:
                        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
                        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
                        os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
                    
                    loader = PlaywrightURLLoader(
                        urls=urls,
                        remove_selectors=["nav", "footer", "script", "style"]
                    )
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
                        Only answer in the context of the program.
                        If contexts don't give you answers explicitly but have helpful links, provide links. Do not say the word "context."
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
                    
                    def inspect(state):
                        """Print the state passed between Runnables in a langchain and pass it on"""
                        print(f'\n\n{state}\n\n')
                        return state
                    
                    # Create RAG chain
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | RunnableLambda(inspect)
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Store in session state
                    st.session_state.rag_chain = rag_chain
                    st.session_state.vectorstore = vectorstore
                    
                    st.success(f"✅ System initialized with {len(splits)} document chunks!")
                    
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")
    
    # Display status
    st.markdown("---")
    if st.session_state.rag_chain is not None:
        st.success("✅ RAG System Ready")
    else:
        st.warning("⚠️ RAG System Not Initialized")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if st.session_state.rag_chain is None:
    st.info("👈 Please configure and initialize the RAG system in the sidebar to start chatting.")
else:
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