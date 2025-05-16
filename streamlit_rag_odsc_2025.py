import os
import sys
from pathlib import Path
from tempfile import mkdtemp

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType, DoclingLoader
from docling.chunking import HybridChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Disable PyTorch MPS warnings
import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported on MPS")

import streamlit as st
# Set page configuration to improve layout
st.set_page_config(
    page_title="PDF Q&A with RAG",
    page_icon="📚",
    layout="wide"
)

# Load environment variables
def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata

        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)

load_dotenv()

# Configuration
HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")

# Find PDF files in current directory if not explicitly provided
DEFAULT_FILE_PATH = "./pdfs/2501.17887v1.pdf"
INPUT_FILE_PATH = os.getenv("PDF_FILE_PATH", DEFAULT_FILE_PATH)

# Check if the file exists, if not search for any PDF
if not os.path.exists(INPUT_FILE_PATH):
    # Search in current directory and subdirectories for PDFs
    pdf_files = []
    for path, subdirs, files in os.walk('.'):
        for name in files:
            if name.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(path, name))
    
    if pdf_files:
        FILE_PATH = [pdf_files[0]]  # Use the first PDF found
        st.sidebar.warning(f"Using found PDF: {pdf_files[0]}")
    else:
        st.error(f"No PDF files found. Please place a PDF file in the project directory.")
        FILE_PATH = []
else:
    FILE_PATH = [INPUT_FILE_PATH]

EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", "microsoft/phi-4")
EXPORT_TYPE = ExportType.DOC_CHUNKS
PROMPT = PromptTemplate.from_template(
    """
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {input}
Answer:
"""
)
TOP_K = int(os.getenv("TOP_K", 3))

# Cache expensive initialization
@st.cache_resource
def init_embeddings():
    # Initialize with proper kwargs to prevent tokenizer warnings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        encode_kwargs={"truncate": True, "max_length": 512},  # Set max sequence length for encoding
        model_kwargs={"device": "cpu"}  # Force CPU usage to avoid MPS issues
    )

@st.cache_resource
def init_llm():
    """
    Initialize a language model client using OpenAI-compatible API.
    Compatible with:
    - OpenAI API
    - vLLM
    - Any OpenAI API-compatible endpoint
    """
    # Get configuration from environment
    openai_api_key = _get_env_from_colab_or_os("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    openai_base_url = os.getenv("OPENAI_API_BASE", None)  # Optional base URL for custom endpoints
    
    # Create ChatOpenAI instance with appropriate configuration
    return ChatOpenAI(
        model_name=openai_model,
        openai_api_key=openai_api_key,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.05")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        openai_api_base=openai_base_url
    )

@st.cache_resource
def init_docling_pipeline():
    if not FILE_PATH:
        return None, None, []
        
    # 1. Load documents via DoclingLoader
    embedding = init_embeddings()
    try:
        # Configure a chunker with appropriate token limits for the embedding model
        custom_chunker = HybridChunker(
            tokenizer=EMBED_MODEL_ID,
            max_tokens=500,  # Set lower than 512 to allow for some buffer
            fallback_method="sentence" 
        )
        
        loader = DoclingLoader(
            file_path=FILE_PATH,
            export_type=EXPORT_TYPE,
            chunker=custom_chunker,
        )
        docs = loader.load()
        
        # 2. Determine splits
        if EXPORT_TYPE == ExportType.DOC_CHUNKS:
            splits = docs
        else:
            from langchain_text_splitters import MarkdownHeaderTextSplitter

            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ('#', 'Header_1'),
                    ('##', 'Header_2'),
                    ('###', 'Header_3'),
                ],
            )
            splits = [chunk for doc in docs for chunk in splitter.split_text(doc.page_content)]

        # 3. Create embeddings and ingest into vector store
        vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)

        # 4. Setup RAG chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        llm = init_llm()
        qa_chain = create_stuff_documents_chain(llm, PROMPT)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        return rag_chain, retriever, splits
    except Exception as e:
        st.error(f"Error in Docling pipeline: {str(e)}")
        return None, None, []

@st.cache_resource
def init_recursive_pipeline():
    if not FILE_PATH:
        return None, None, []
        
    try:
        # 1. Load documents via PyPDFLoader
        embedding = init_embeddings()
        docs = []
        for file in FILE_PATH:
            loader = PyPDFLoader(file)
            docs.extend(loader.load())
        
        # 2. Split documents using RecursiveCharacterTextSplitter
        # Use smaller chunk size and overlap to avoid token length issues
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced from 1000
            chunk_overlap=0,  # Reduced from 200
        )
        splits = text_splitter.split_documents(docs)

        # 3. Create embeddings and ingest into vector store
        vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)

        # 4. Setup RAG chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        llm = init_llm()
        qa_chain = create_stuff_documents_chain(llm, PROMPT)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        return rag_chain, retriever, splits
    except Exception as e:
        st.error(f"Error in recursive pipeline: {str(e)}")
        return None, None, []

# Function to display retrieved documents with color coding
def display_retrieved_docs(retriever, query, title, border_color=None):
    if retriever is None:
        st.warning(f"{title}: Retriever not available")
        return []
    
    # Default border color if none provided
    if border_color is None:
        border_color = '#4CAF50'  # Default green border
        
    try:
        docs = retriever.get_relevant_documents(query)
        st.subheader(title)
        for i, doc in enumerate(docs):
            # Use styled divs instead of expanders to avoid nesting issues
            st.markdown(
                f"""
                <div style="border-left: 5px solid {border_color}; padding: 10px 15px; margin-bottom: 10px;">
                <p><strong>Document {i+1}</strong></p>
                <p><strong>Source</strong>: {doc.metadata.get('source', 'Unknown')}</p>
                <p><strong>Page</strong>: {doc.metadata.get('page', 'Unknown')}</p>
                <p><strong>Content</strong>:</p>
                <div style="margin-top: 10px;">{doc.page_content}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        return docs
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

# Initialize Streamlit app with proper async handling
def main():
    st.title("📚 PDF Q&A with RAG - Chunker Comparison")
    st.markdown("Compare document retrieval using Docling's HybridChunker vs RecursiveCharacterTextSplitter")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    if FILE_PATH:
        st.sidebar.success(f"Using PDF: {FILE_PATH[0]}")
    else:
        st.sidebar.error("No PDF files found")
        st.stop()
    
    st.sidebar.info(f"Using vector store: FAISS")
    st.sidebar.info(f"Using embedding model: {EMBED_MODEL_ID}")
    openai_model = os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')
    openai_base_url = os.getenv('OPENAI_API_BASE', 'default OpenAI API')
    base_url_display = "Custom Endpoint" if openai_base_url and openai_base_url != 'default OpenAI API' else "Official OpenAI API"
    model_info = f"Using LLM: {openai_model} via {base_url_display}"
    st.sidebar.info(model_info)

    # Initialize or retrieve chains
    with st.spinner("Loading document processing pipelines..."):
        docling_chain, docling_retriever, docling_splits = init_docling_pipeline()
        recursive_chain, recursive_retriever, recursive_splits = init_recursive_pipeline()
    
    if not docling_splits or not recursive_splits:
        st.error("Error initializing document splits. Please check file paths and try again.")
        return
    
    # Display statistics about chunks
    with st.expander("Chunking Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("HybridChunker")
            st.metric("Number of chunks", len(docling_splits))
            
            # Calculate average chunk length
            if docling_splits:
                avg_length = sum(len(doc.page_content) for doc in docling_splits) / len(docling_splits)
                st.metric("Avg. chunk length (chars)", f"{avg_length:.0f}")
                
                # Calculate token distribution if available
                token_counts = [doc.metadata.get('token_count', 0) for doc in docling_splits if 'token_count' in doc.metadata]
                if token_counts:
                    st.metric("Avg. tokens per chunk", f"{sum(token_counts)/len(token_counts):.0f}")
        
        with col2:
            st.subheader("RecursiveCharacterTextSplitter")
            st.metric("Number of chunks", len(recursive_splits))
            
            # Calculate average chunk length
            if recursive_splits:
                avg_length = sum(len(doc.page_content) for doc in recursive_splits) / len(recursive_splits)
                st.metric("Avg. chunk length (chars)", f"{avg_length:.0f}")

    # Handle user input
    if user_input := st.chat_input("Ask anything about the PDF..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check if we have retrievers
        if not docling_retriever or not recursive_retriever:
            st.error("One or both retrievers failed to initialize. Please check the logs for details.")
            return
            
        # Run RAG chain with hybrid chunker (primary response)
        try:
            with st.spinner("Generating answer using HybridChunker..."):
                resp = docling_chain.invoke({"input": user_input})
            answer = resp.get('answer', '').strip()
        except Exception as e:
            st.error(f"Error generating HybridChunker answer: {str(e)}")
            answer = "Error generating answer. Please try again or check configuration."
        
        # Run comparison with RecursiveCharacterTextSplitter
        try:
            with st.spinner("Generating comparison answer using RecursiveCharacterTextSplitter..."):
                recursive_resp = recursive_chain.invoke({"input": user_input})
            recursive_answer = recursive_resp.get('answer', '').strip()
        except Exception as e:
            st.error(f"Error generating RecursiveCharacterTextSplitter answer: {str(e)}")
            recursive_answer = "Error generating comparison answer. Please try again or check configuration."
        
        # Display debug information first
        st.header("Debug Information")

        # Create expanders for debug information - FIXED: No nested expanders
        with st.expander("🔍 HybridChunker Debug Information", expanded=False):
            st.subheader("Full Prompt")
            retrieved_docs = docling_retriever.get_relevant_documents(user_input)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            full_prompt = PROMPT.format(context=context_text, input=user_input)
            st.code(full_prompt, language="markdown")

            # Display documents without nested expanders
            st.subheader("Documents Used in Context:")
            for i, doc in enumerate(retrieved_docs):
                # Use a styled div instead of nested expanders
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                    <h4>Document {i+1}</h4>
                    <p><strong>Source</strong>: {doc.metadata.get('source', 'Unknown')}</p>
                    <p><strong>Page</strong>: {doc.metadata.get('page', 'Unknown')}</p>
                    {f"<p><strong>Chunk ID</strong>: {doc.metadata['chunk_id']}</p>" if 'chunk_id' in doc.metadata else ""}
                    <p><strong>Content</strong>:</p>
                    <pre>{doc.page_content}</pre>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with st.expander("🔍 RecursiveCharacterTextSplitter Debug Information", expanded=False):
            st.subheader("Full Prompt")
            retrieved_docs = recursive_retriever.get_relevant_documents(user_input)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            full_prompt = PROMPT.format(context=context_text, input=user_input)
            st.code(full_prompt, language="markdown")

            # Display documents without nested expanders
            st.subheader("Documents Used in Context:")
            for i, doc in enumerate(retrieved_docs):
                # Use a styled div instead of nested expanders
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                    <h4>Document {i+1}</h4>
                    <p><strong>Source</strong>: {doc.metadata.get('source', 'Unknown')}</p>
                    <p><strong>Page</strong>: {doc.metadata.get('page', 'Unknown')}</p>
                    {f"<p><strong>Chunk ID</strong>: {doc.metadata['chunk_id']}</p>" if 'chunk_id' in doc.metadata else ""}
                    <p><strong>Content</strong>:</p>
                    <pre>{doc.page_content}</pre>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Show retrieved documents comparison
        st.header("Retrieved Documents Comparison")
        
        # Use expanders for document comparison sections
        with st.expander("HybridChunker Results", expanded=False):
            docling_docs = display_retrieved_docs(
                docling_retriever, 
                user_input, 
                "Retrieved Documents",
                border_color="#4CAF50"  # Green border
            )
            
        with st.expander("RecursiveCharacterTextSplitter Results", expanded=False):
            recursive_docs = display_retrieved_docs(
                recursive_retriever, 
                user_input, 
                "Retrieved Documents",
                border_color="#f44336"  # Red border
            )
        
        # Show answers sequentially
        st.header("Answer Comparison")
        
        # First answer with colored border
        st.subheader("🔍 HybridChunker Answer")
        st.markdown(
            f"""
            <div style="border-left: 5px solid #4CAF50; padding: 15px; margin-bottom: 20px;">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Second answer with different colored border
        st.subheader("🔍 RecursiveCharacterTextSplitter Answer")
        st.markdown(
            f"""
            <div style="border-left: 5px solid #f44336; padding: 15px; margin-bottom: 20px;">
            {recursive_answer}
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    # Handle asyncio errors more gracefully
    import asyncio
    try:
        # For macOS - ensure a new event loop is created
        if sys.platform == 'darwin':
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        main()
    except RuntimeError as e:
        if "no running event loop" in str(e):
            # Create a new event loop if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            main()
        else:
            st.error(f"Runtime Error: {str(e)}")
    except Exception as e:
        st.error(f"Error running the application: {str(e)}")
        st.error("Trace:")
        import traceback
        st.code(traceback.format_exc())
