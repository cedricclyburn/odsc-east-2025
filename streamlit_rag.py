import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
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
from pydantic import Field, validator
from pydantic_settings import BaseSettings
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Disable PyTorch MPS warnings
import warnings

warnings.filterwarnings(
    "ignore", message="'pin_memory' argument is set as true but not supported on MPS"
)

import streamlit as st

# Set page configuration to improve layout
st.set_page_config(page_title="PDF Q&A with RAG", page_icon="📚", layout="wide")


class ChunkerType(str, Enum):
    """Enum for different chunker types"""

    HYBRID = "HybridChunker"
    RECURSIVE = "RecursiveCharacterTextSplitter"


class AppSettings(BaseSettings):
    """Application settings using pydantic for validation"""

    # API keys and tokens
    hf_token: Optional[str] = Field(None, env="HF_TOKEN")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # Model settings
    embed_model_id: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", env="EMBED_MODEL_ID"
    )
    openai_model: str = Field("llama3.2:3b", env="OPENAI_MODEL_NAME")
    openai_base_url: Optional[str] = Field(
        "http://localhost:11434/v1", env="OPENAI_API_BASE"
    )

    # RAG parameters
    top_k: int = Field(3, env="TOP_K")
    llm_temperature: float = Field(0.05, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(1024, env="LLM_MAX_TOKENS")

    # Chunking parameters
    hybrid_max_tokens: int = Field(500, env="HYBRID_MAX_TOKENS")
    recursive_chunk_size: int = Field(500, env="RECURSIVE_CHUNK_SIZE")
    recursive_chunk_overlap: int = Field(50, env="RECURSIVE_CHUNK_OVERLAP")

    # File paths
    pdf_file_path: Optional[str] = Field(None, env="PDF_FILE_PATH")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields

    @validator("top_k")
    def validate_top_k(cls, v):
        if v < 1:
            return 1
        return v


class RAGPipeline:
    """Base class for RAG pipelines"""

    def __init__(self, settings: AppSettings, embedding_model: HuggingFaceEmbeddings):
        self.settings = settings
        self.embedding_model = embedding_model
        self.llm = self._init_llm()
        self.prompt = PromptTemplate.from_template(
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

    def _init_llm(self):
        """Initialize a language model client using OpenAI-compatible API"""
        # If using a custom API endpoint with a model like Phi-4, we need to use that model name
        if (
            self.settings.openai_base_url
            and "phi-4" in self.settings.openai_base_url.lower()
        ):
            # This looks like a Phi-4 deployment - use a compatible model name
            model_name = "microsoft/phi-4"
        else:
            # Use the configured model name
            model_name = self.settings.openai_model

        try:
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=self.settings.openai_api_key,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                openai_api_base=self.settings.openai_base_url,
            )
        except Exception as e:
            st.error(f"Error initializing LLM with model '{model_name}': {str(e)}")
            # Fallback to Phi-4 naming only for Phi deployments
            if (
                self.settings.openai_base_url
                and "phi" in self.settings.openai_base_url.lower()
            ):
                st.warning(
                    "Trying fallback model name 'microsoft/phi-4' with custom endpoint"
                )
                return ChatOpenAI(
                    model_name="microsoft/phi-4",  # Azure OpenAI uses this format
                    openai_api_key=self.settings.openai_api_key,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_tokens,
                    openai_api_base=self.settings.openai_base_url,
                )
            raise

    def process_files(
        self, file_paths: List[str]
    ) -> Tuple[Optional[Any], Optional[BaseRetriever], List[Document]]:
        """
        Abstract method to process files and create a RAG pipeline

        Returns:
            Tuple[Optional[Any], Optional[BaseRetriever], List[Document]]:
                The RAG chain, retriever, and document splits
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _create_rag_chain(self, splits: List[Document]) -> Tuple[Any, BaseRetriever]:
        """Create a RAG chain from document splits"""
        vectorstore = FAISS.from_documents(
            documents=splits, embedding=self.embedding_model
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.settings.top_k})
        qa_chain = create_stuff_documents_chain(self.llm, self.prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        return rag_chain, retriever


class DoclingRAGPipeline(RAGPipeline):
    """RAG pipeline implementation using DoclingLoader with HybridChunker"""

    def process_files(
        self, file_paths: List[str]
    ) -> Tuple[Optional[Any], Optional[BaseRetriever], List[Document]]:
        if not file_paths:
            return None, None, []

        try:
            # Configure a chunker with appropriate token limits for the embedding model
            custom_chunker = HybridChunker(
                tokenizer=self.settings.embed_model_id,
                max_tokens=self.settings.hybrid_max_tokens,
                fallback_method="sentence",
            )

            loader = DoclingLoader(
                file_path=file_paths,
                export_type=ExportType.DOC_CHUNKS,
                chunker=custom_chunker,
            )
            docs = loader.load()

            # Create RAG chain
            rag_chain, retriever = self._create_rag_chain(docs)

            return rag_chain, retriever, docs
        except Exception as e:
            st.error(f"Error in Docling pipeline: {str(e)}")
            return None, None, []


class RecursiveRAGPipeline(RAGPipeline):
    """RAG pipeline implementation using PyPDFLoader with RecursiveCharacterTextSplitter"""

    def process_files(
        self, file_paths: List[str]
    ) -> Tuple[Optional[Any], Optional[BaseRetriever], List[Document]]:
        if not file_paths:
            return None, None, []

        try:
            # Load documents via PyPDFLoader
            docs = []
            for file in file_paths:
                loader = PyPDFLoader(file)
                docs.extend(loader.load())

            # Split documents using RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.recursive_chunk_size,
                chunk_overlap=self.settings.recursive_chunk_overlap,
            )
            splits = text_splitter.split_documents(docs)

            # Create RAG chain
            rag_chain, retriever = self._create_rag_chain(splits)

            return rag_chain, retriever, splits
        except Exception as e:
            st.error(f"Error in recursive pipeline: {str(e)}")
            return None, None, []


class RAGPipelineFactory:
    """Factory class for creating RAG pipelines"""

    @staticmethod
    def create_pipeline(
        pipeline_type: ChunkerType,
        settings: AppSettings,
        embedding_model: HuggingFaceEmbeddings,
    ) -> RAGPipeline:
        """Create a RAG pipeline based on the specified type"""
        if pipeline_type == ChunkerType.HYBRID:
            return DoclingRAGPipeline(settings, embedding_model)
        elif pipeline_type == ChunkerType.RECURSIVE:
            return RecursiveRAGPipeline(settings, embedding_model)
        else:
            raise ValueError(f"Unsupported pipeline type: {pipeline_type}")


class FileManager:
    """Class to handle file operations"""

    @staticmethod
    def find_pdf_files() -> List[str]:
        """Find PDF files in the current directory and subdirectories"""
        pdf_files = []
        for path, subdirs, files in os.walk("."):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(path, name))
        return pdf_files

    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """Save an uploaded file to a temporary directory"""
        temp_dir = mkdtemp()
        temp_path = Path(temp_dir) / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(temp_path)


class DocumentComparison:
    """Class to handle document comparison and metrics"""

    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self.embedding_model = embedding_model

    def compute_similarity(self, query: str, docs: List[Document]) -> List[float]:
        """Compute similarity scores between query and documents"""
        if not docs:
            return []

        # Embed query and documents
        query_embedding = self.embedding_model.embed_query(query)
        doc_embeddings = [
            self.embedding_model.embed_query(doc.page_content) for doc in docs
        ]

        # Compute cosine similarity
        similarities = []
        for doc_embedding in doc_embeddings:
            query_embed = np.array(query_embedding).reshape(1, -1)
            doc_embed = np.array(doc_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_embed, doc_embed)[0][0]
            similarities.append(float(similarity))

        return similarities

    def create_token_distribution_chart(
        self, docling_splits: List[Document], recursive_splits: List[Document]
    ) -> plt.Figure:
        """Create a chart comparing token distribution between chunking methods"""
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Extract token counts if available, otherwise estimate
        docling_tokens = []
        for doc in docling_splits:
            if "token_count" in doc.metadata:
                docling_tokens.append(doc.metadata["token_count"])
            else:
                # Rough estimate - 4 chars per token
                docling_tokens.append(len(doc.page_content) // 4)

        recursive_tokens = [
            len(doc.page_content) // 4 for doc in recursive_splits
        ]  # Rough estimate

        # Create histograms
        ax[0].hist(docling_tokens, bins=20, alpha=0.7, color="green")
        ax[0].set_title("HybridChunker Token Distribution")
        ax[0].set_xlabel("Token Count")
        ax[0].set_ylabel("Number of Chunks")

        ax[1].hist(recursive_tokens, bins=20, alpha=0.7, color="red")
        ax[1].set_title("RecursiveCharacterTextSplitter Token Distribution")
        ax[1].set_xlabel("Token Count")
        ax[1].set_ylabel("Number of Chunks")

        plt.tight_layout()
        return fig


class ChatHistory:
    """Class to manage chat history"""

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def add_user_message(self, message: str):
        """Add a user message to the chat history"""
        st.session_state.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        """Add an assistant message to the chat history"""
        st.session_state.messages.append({"role": "assistant", "content": message})

    def display_messages(self):
        """Display all messages in the chat history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


class PerformanceMetrics:
    """Class to track and display performance metrics"""

    def __init__(self):
        self.timings = {}

    def start_timer(self, name: str):
        """Start a timer for a specific operation"""
        self.timings[name] = {"start": time.time(), "end": None, "duration": None}

    def end_timer(self, name: str):
        """End a timer for a specific operation"""
        if name in self.timings:
            self.timings[name]["end"] = time.time()
            self.timings[name]["duration"] = (
                self.timings[name]["end"] - self.timings[name]["start"]
            )

    def get_duration(self, name: str) -> Optional[float]:
        """Get the duration of a specific operation"""
        if name in self.timings and self.timings[name]["duration"] is not None:
            return self.timings[name]["duration"]
        return None

    def display_metrics(self):
        """Display all performance metrics"""
        metrics = {
            name: timing["duration"]
            for name, timing in self.timings.items()
            if timing["duration"] is not None
        }
        if metrics:
            st.subheader("Performance Metrics")
            for name, duration in metrics.items():
                st.metric(name, f"{duration:.3f}s")


class PDFQAApp:
    """Main application class"""

    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize settings
        self.settings = AppSettings()

        # Initialize components
        self.file_manager = FileManager()
        self.chat_history = ChatHistory()
        self.performance = PerformanceMetrics()

        # State variables
        self.file_paths = []
        self.embedding_model = None
        self.docling_pipeline = None
        self.recursive_pipeline = None
        self.docling_chain = None
        self.docling_retriever = None
        self.docling_splits = []
        self.recursive_chain = None
        self.recursive_retriever = None
        self.recursive_splits = []
        self.document_comparison = None

    def initialize_components(self):
        """Initialize all components needed for the application"""
        # Initialize embedding model
        self.embedding_model = self._init_embeddings()

        # Initialize document comparison
        self.document_comparison = DocumentComparison(self.embedding_model)

        # Initialize pipelines if files are available
        if self.file_paths:
            # Create pipelines
            self.docling_pipeline = RAGPipelineFactory.create_pipeline(
                ChunkerType.HYBRID, self.settings, self.embedding_model
            )
            self.recursive_pipeline = RAGPipelineFactory.create_pipeline(
                ChunkerType.RECURSIVE, self.settings, self.embedding_model
            )

            # Process files
            self.performance.start_timer("HybridChunker Processing")
            self.docling_chain, self.docling_retriever, self.docling_splits = (
                self.docling_pipeline.process_files(self.file_paths)
            )
            self.performance.end_timer("HybridChunker Processing")

            self.performance.start_timer("RecursiveCharacterTextSplitter Processing")
            self.recursive_chain, self.recursive_retriever, self.recursive_splits = (
                self.recursive_pipeline.process_files(self.file_paths)
            )
            self.performance.end_timer("RecursiveCharacterTextSplitter Processing")

    def _init_embeddings(self):
        """Initialize the embedding model"""
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            # Fall back to CPU if torch is unavailable or device check fails
            device = "cpu"

        return HuggingFaceEmbeddings(
            model_name=self.settings.embed_model_id, model_kwargs={"device": device}
        )

    def setup_sidebar(self):
        """Set up the sidebar with configuration options"""
        st.sidebar.title("Configuration")

        # File upload
        uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
        if uploaded_file:
            file_path = self.file_manager.save_uploaded_file(uploaded_file)
            self.file_paths = [file_path]
            st.sidebar.success(f"Using uploaded PDF: {uploaded_file.name}")
        else:
            # If no file uploaded, look for default or find PDFs
            default_file_path = self.settings.pdf_file_path
            if default_file_path and os.path.exists(default_file_path):
                self.file_paths = [default_file_path]
                st.sidebar.success(f"Using default PDF: {default_file_path}")
            else:
                pdf_files = self.file_manager.find_pdf_files()
                if pdf_files:
                    selected_pdf = st.sidebar.selectbox(
                        "Select a PDF file",
                        options=pdf_files,
                        format_func=lambda x: os.path.basename(x),
                    )
                    self.file_paths = [selected_pdf]
                    st.sidebar.success(
                        f"Using found PDF: {os.path.basename(selected_pdf)}"
                    )
                else:
                    st.sidebar.error("No PDF files found. Please upload a PDF file.")

        # Display current settings
        st.sidebar.subheader("Current Settings")
        st.sidebar.info(f"Vector store: FAISS")
        st.sidebar.info(f"Embedding model: {self.settings.embed_model_id}")

        base_url_display = (
            "Custom Endpoint"
            if self.settings.openai_base_url
            else "Official OpenAI API"
        )
        model_info = f"LLM: {self.settings.openai_model} via {base_url_display}"
        st.sidebar.info(model_info)

        # Chunking parameters
        st.sidebar.subheader("Chunking Parameters")

        # HybridChunker parameters
        hybrid_max_tokens = st.sidebar.slider(
            "HybridChunker Max Tokens",
            min_value=100,
            max_value=1000,
            value=self.settings.hybrid_max_tokens,
            step=50,
        )
        if hybrid_max_tokens != self.settings.hybrid_max_tokens:
            self.settings.hybrid_max_tokens = hybrid_max_tokens
            st.sidebar.warning(
                "Changing this parameter requires reprocessing the documents."
            )
            st.sidebar.button(
                "Reprocess Documents", key="hybrid", on_click=self.initialize_components
            )

        # RecursiveCharacterTextSplitter parameters
        recursive_chunk_size = st.sidebar.slider(
            "RecursiveCharacterTextSplitter Chunk Size",
            min_value=100,
            max_value=1000,
            value=self.settings.recursive_chunk_size,
            step=50,
        )
        recursive_chunk_overlap = st.sidebar.slider(
            "RecursiveCharacterTextSplitter Chunk Overlap",
            min_value=0,
            max_value=200,
            value=self.settings.recursive_chunk_overlap,
            step=10,
        )
        if (
            recursive_chunk_size != self.settings.recursive_chunk_size
            or recursive_chunk_overlap != self.settings.recursive_chunk_overlap
        ):
            self.settings.recursive_chunk_size = recursive_chunk_size
            self.settings.recursive_chunk_overlap = recursive_chunk_overlap
            st.sidebar.warning(
                "Changing these parameters requires reprocessing the documents."
            )
            st.sidebar.button(
                "Reprocess Documents",
                key="recursive",
                on_click=self.initialize_components,
            )

        # RAG parameters
        st.sidebar.subheader("RAG Parameters")
        top_k = st.sidebar.slider(
            "Number of Retrieved Documents (Top K)",
            min_value=1,
            max_value=10,
            value=self.settings.top_k,
        )
        self.settings.top_k = top_k

    def display_chunking_statistics(self):
        """Display statistics about the document chunks"""
        if not self.docling_splits and not self.recursive_splits:
            st.warning("No document splits available for comparison.")
            return

        with st.expander("Chunking Statistics", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("HybridChunker")
                st.metric("Number of chunks", len(self.docling_splits))

                # Calculate average chunk length
                if self.docling_splits:
                    avg_length = sum(
                        len(doc.page_content) for doc in self.docling_splits
                    ) / len(self.docling_splits)
                    st.metric("Avg. chunk length (chars)", f"{avg_length:.0f}")

                    # Calculate token distribution if available
                    token_counts = [
                        doc.metadata.get("token_count", 0)
                        for doc in self.docling_splits
                        if "token_count" in doc.metadata
                    ]
                    if token_counts:
                        st.metric(
                            "Avg. tokens per chunk",
                            f"{sum(token_counts) / len(token_counts):.0f}",
                        )
                        st.metric("Min tokens", min(token_counts))
                        st.metric("Max tokens", max(token_counts))

            with col2:
                st.subheader("RecursiveCharacterTextSplitter")
                st.metric("Number of chunks", len(self.recursive_splits))

                # Calculate average chunk length
                if self.recursive_splits:
                    avg_length = sum(
                        len(doc.page_content) for doc in self.recursive_splits
                    ) / len(self.recursive_splits)
                    st.metric("Avg. chunk length (chars)", f"{avg_length:.0f}")

                    # Estimate tokens based on character count (rough estimate)
                    estimated_tokens = [
                        len(doc.page_content) // 4 for doc in self.recursive_splits
                    ]
                    if estimated_tokens:
                        st.metric(
                            "Est. avg. tokens per chunk",
                            f"{sum(estimated_tokens) / len(estimated_tokens):.0f}",
                        )
                        st.metric("Est. min tokens", min(estimated_tokens))
                        st.metric("Est. max tokens", max(estimated_tokens))

            # Add token distribution visualization
            if self.docling_splits and self.recursive_splits:
                st.subheader("Token Distribution Comparison")
                fig = self.document_comparison.create_token_distribution_chart(
                    self.docling_splits, self.recursive_splits
                )
                st.pyplot(fig)

    def display_retrieved_docs(self, retriever, query, title, border_color=None):
        """Display retrieved documents with color coding and similarity scores"""
        if retriever is None:
            st.warning(f"{title}: Retriever not available")
            return []

        # Default border color if none provided
        if border_color is None:
            border_color = "#4CAF50"  # Default green border

        try:
            # Get documents and compute similarity scores
            self.performance.start_timer(f"{title} Retrieval")
            docs = retriever.get_relevant_documents(query)
            self.performance.end_timer(f"{title} Retrieval")

            similarities = self.document_comparison.compute_similarity(query, docs)

            st.subheader(title)
            for i, (doc, similarity) in enumerate(zip(docs, similarities)):
                # Use styled divs with similarity score
                st.markdown(
                    f"""
                    <div style="border: 1px solid {border_color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                    <div><strong>Document {i + 1}</strong></div>
                    <div style="padding-left: 1em;"><strong>Similarity </strong>: {similarity:.4f}</div>
                    <div style="padding-left: 1em;"><strong>Source </strong>: {doc.metadata.get("source", "Unknown")}</div>
                    <div style="padding-left: 1em;"><strong>Page </strong>: {doc.metadata.get("page", "Unknown")}</div>
                    <div style="padding-left: 1em;"><strong>Chunk ID </strong>: {doc.metadata.get("chunk_id", "Unknown")}</div>
                    <div><strong>Content </strong>:</div>
                    <div>{doc.page_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            return docs
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []

    def process_user_input(self, user_input):
        """Process user input and generate responses"""
        # Display user message
        self.chat_history.add_user_message(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        # Check if we have retrievers
        if not self.docling_retriever or not self.recursive_retriever:
            st.error(
                "One or both retrievers failed to initialize. Please check the logs for details."
            )
            return

        # Run RAG chains
        try:
            # HybridChunker response
            self.performance.start_timer("HybridChunker Answer Generation")
            docling_resp = self.docling_chain.invoke({"input": user_input})
            self.performance.end_timer("HybridChunker Answer Generation")
            docling_answer = docling_resp.get("answer", "").strip()

            # RecursiveCharacterTextSplitter response
            self.performance.start_timer(
                "RecursiveCharacterTextSplitter Answer Generation"
            )
            recursive_resp = self.recursive_chain.invoke({"input": user_input})
            self.performance.end_timer(
                "RecursiveCharacterTextSplitter Answer Generation"
            )
            recursive_answer = recursive_resp.get("answer", "").strip()

            # Create combined response for chat history
            combined_response = f"""
            ## 🔍 Comparison Results

            ### HybridChunker Answer
            {docling_answer}
            
            ### RecursiveCharacterTextSplitter Answer
            {recursive_answer}
            """
            self.chat_history.add_assistant_message(combined_response)

            # Display debug information
            st.header("Debug Information")

            # Display performance metrics
            with st.expander("🔍 Debug Information", expanded=False):
                self.performance.display_metrics()

            # Create expanders for debug information
            with st.expander("🔍 HybridChunker Debug Information", expanded=False):
                st.subheader("Full Prompt")
                retrieved_docs = self.docling_retriever.get_relevant_documents(
                    user_input
                )
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                full_prompt = self.docling_pipeline.prompt.format(
                    context=context_text, input=user_input
                )
                st.code(full_prompt, language="markdown")

                # Display documents without nested expanders
                st.subheader("Documents Used in Context:")
                for i, doc in enumerate(retrieved_docs):
                    # Use a styled div instead of nested expanders
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <h4>Document {i + 1}</h4>
                        <p><strong>Source</strong>: {doc.metadata.get("source", "Unknown")}</p>
                        <p><strong>Page</strong>: {doc.metadata.get("page", "Unknown")}</p>
                        {f"<p><strong>Chunk ID</strong>: {doc.metadata['chunk_id']}</p>" if "chunk_id" in doc.metadata else ""}
                        <p><strong>Content</strong>:</p>
                        <pre>{doc.page_content}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander(
                "🔍 RecursiveCharacterTextSplitter Debug Information", expanded=False
            ):
                st.subheader("Full Prompt")
                retrieved_docs = self.recursive_retriever.get_relevant_documents(
                    user_input
                )
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                full_prompt = self.recursive_pipeline.prompt.format(
                    context=context_text, input=user_input
                )
                st.code(full_prompt, language="markdown")

                # Display documents without nested expanders
                st.subheader("Documents Used in Context:")
                for i, doc in enumerate(retrieved_docs):
                    # Use a styled div instead of nested expanders
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <h4>Document {i + 1}</h4>
                        <p><strong>Source</strong>: {doc.metadata.get("source", "Unknown")}</p>
                        <p><strong>Page</strong>: {doc.metadata.get("page", "Unknown")}</p>
                        {f"<p><strong>Chunk ID</strong>: {doc.metadata['chunk_id']}</p>" if "chunk_id" in doc.metadata else ""}
                        <p><strong>Content</strong>:</p>
                        <pre>{doc.page_content}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # # Show retrieved documents comparison
            # st.header("Retrieved Documents Comparison")
            ###
            # Display retrieved documents
            with st.expander("Retrieved Documents Comparison", expanded=False):
                import re
                from collections import Counter

                col1, col2 = st.columns(2)

                with col1:
                    docling_docs = self.display_retrieved_docs(
                        self.docling_retriever,
                        user_input,
                        "HybridChunker Documents",
                        border_color="#4CAF50",
                    )

                with col2:
                    recursive_docs = self.display_retrieved_docs(
                        self.recursive_retriever,
                        user_input,
                        "RecursiveCharacterTextSplitter Documents",
                        border_color="#f44336",
                    )

                # Add document similarity comparison
                if docling_docs and recursive_docs:
                    st.subheader("Document Similarity Comparison")
                    docling_similarities = self.document_comparison.compute_similarity(
                        user_input, docling_docs
                    )
                    recursive_similarities = (
                        self.document_comparison.compute_similarity(
                            user_input, recursive_docs
                        )
                    )

                    # Create a bar chart comparing similarities
                    fig, ax = plt.subplots(figsize=(10, 5))
                    x = np.arange(
                        min(len(docling_similarities), len(recursive_similarities))
                    )
                    width = 0.35

                    if x.size > 0:  # Check if there are documents to compare
                        rects1 = ax.bar(
                            x - width / 2,
                            docling_similarities[: len(x)],
                            width,
                            label="HybridChunker",
                            color="green",
                            alpha=0.7,
                        )
                        rects2 = ax.bar(
                            x + width / 2,
                            recursive_similarities[: len(x)],
                            width,
                            label="RecursiveCharacterTextSplitter",
                            color="red",
                            alpha=0.7,
                        )

                        ax.set_ylabel("Similarity Score")
                        ax.set_title("Query-Document Similarity Comparison")
                        ax.set_xticks(x)
                        ax.set_xticklabels([f"Doc {i + 1}" for i in range(len(x))])
                        ax.legend()
                        ax.set_ylim(0, 1)

                        # Add labels on top of bars
                        def autolabel(rects):
                            for rect in rects:
                                height = rect.get_height()
                                ax.annotate(
                                    f"{height:.3f}",
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha="center",
                                    va="bottom",
                                    rotation=0,
                                )

                        autolabel(rects1)
                        autolabel(rects2)

                        plt.tight_layout()
                        st.pyplot(fig)

            # Show answers sequentially
            st.header("Answer Comparison")

            # First answer with colored border
            st.subheader("🔍 HybridChunker Answer")
            st.markdown(
                f"""
                <div style="border-left: 5px solid #4CAF50; padding: 15px; margin-bottom: 20px;">
                {docling_answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Second answer with different colored border
            st.subheader("🔍 RecursiveCharacterTextSplitter Answer")
            st.markdown(
                f"""
                <div style="border-left: 5px solid #f44336; padding: 15px; margin-bottom: 20px;">
                {recursive_answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # # Highlight matching key phrases
            # with st.expander("Retrieved Documents Comparison", expanded=False):
            #     st.subheader("Retrieved Documents")

            #     # Extract key phrases from the query (simple approach)
            #     import re
            #     from collections import Counter

            #     col1, col2 = st.columns(2)

            #     with col1:
            #         st.markdown("### HybridChunker Documents with Highlights")
            #         for i, doc in enumerate(docling_docs):
            #             st.markdown(
            #                 f"""
            #                 <div style="border: 1px solid #4CAF50; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            #                 <p><strong>Document {i+1}</strong></p>
            #                 <div>{doc.page_content}</div>
            #                 </div>
            #                 """,
            #                 unsafe_allow_html=True
            #             )

            #     with col2:
            #         st.markdown("### RecursiveCharacterTextSplitter Documents with Highlights")
            #         for i, doc in enumerate(recursive_docs):
            #             st.markdown(
            #                 f"""
            #                 <div style="border: 1px solid #f44336; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
            #                 <p><strong>Document {i+1}</strong></p>
            #                 <div>{doc.page_content}</div>
            #                 </div>
            #                 """,
            #                 unsafe_allow_html=True
            #             )
        except Exception as e:
            st.error(f"Error generating answers: {str(e)}")
            import traceback

            st.error(traceback.format_exc())

    def export_results(self):
        """Create an export feature for comparison results"""
        with st.expander("Export Comparison Results", expanded=False):
            st.subheader("Export Options")

            if not self.docling_splits or not self.recursive_splits:
                st.warning("No results available for export.")
                return

            # Create downloadable content
            import io
            import zipfile

            # Create a buffer for the zip file
            buffer = io.BytesIO()

            # Create markdown export of chunking statistics
            chunking_stats = f"""# Chunking Statistics Comparison
            
## HybridChunker
- Number of chunks: {len(self.docling_splits)}
- Avg. chunk length (chars): {sum(len(doc.page_content) for doc in self.docling_splits) / len(self.docling_splits):.0f}

## RecursiveCharacterTextSplitter
- Number of chunks: {len(self.recursive_splits)}
- Avg. chunk length (chars): {sum(len(doc.page_content) for doc in self.recursive_splits) / len(self.recursive_splits):.0f}
            """

            # Add performance metrics if available
            if self.performance.timings:
                chunking_stats += "\n\n## Performance Metrics\n"
                for name, timing in self.performance.timings.items():
                    if timing["duration"] is not None:
                        chunking_stats += f"- {name}: {timing['duration']:.3f}s\n"

            st.download_button(
                label="Download Stats as Markdown",
                data=chunking_stats,
                file_name="chunking_stats.md",
                mime="text/markdown",
            )

            # Create CSV export of chunk data
            csv_buffer = io.StringIO()
            csv_buffer.write("Chunker,ChunkID,PageNum,CharCount,EstTokenCount\n")

            for doc in self.docling_splits:
                chunk_id = doc.metadata.get("chunk_id", "N/A")
                page = doc.metadata.get("page", "N/A")
                char_count = len(doc.page_content)
                token_count = doc.metadata.get(
                    "token_count", char_count // 4
                )  # Use actual or estimate
                csv_buffer.write(
                    f"HybridChunker,{chunk_id},{page},{char_count},{token_count}\n"
                )

            for doc in self.recursive_splits:
                chunk_id = (
                    "N/A"  # RecursiveCharacterTextSplitter doesn't assign chunk IDs
                )
                page = doc.metadata.get("page", "N/A")
                char_count = len(doc.page_content)
                token_count = char_count // 4  # Estimate
                csv_buffer.write(
                    f"RecursiveCharacterTextSplitter,{chunk_id},{page},{char_count},{token_count}\n"
                )

            st.download_button(
                label="Download Chunk Data as CSV",
                data=csv_buffer.getvalue(),
                file_name="chunk_data.csv",
                mime="text/csv",
            )

    def run(self):
        """Run the Streamlit application"""
        st.title("📚 PDF Q&A with RAG - Chunker Comparison")
        st.markdown(
            "Compare document retrieval using Docling's HybridChunker vs RecursiveCharacterTextSplitter"
        )

        # Setup sidebar and get file paths
        self.setup_sidebar()

        # Initialize components
        try:
            with st.spinner("Loading document processing pipelines..."):
                self.initialize_components()
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            st.error("Check your environment variables and API credentials.")
            import traceback

            st.code(traceback.format_exc())
            st.warning("The application will continue with limited functionality.")

        # Display chunking statistics
        self.display_chunking_statistics()

        # Add export functionality
        self.export_results()

        # Display previous messages
        self.chat_history.display_messages()

        # Handle user input
        if user_input := st.chat_input("Ask anything about the PDF..."):
            if not self.file_paths:
                st.error("Please upload or select a PDF file first.")
                return

            if not self.docling_retriever or not self.recursive_retriever:
                st.error(
                    "RAG components failed to initialize. Please check your configuration."
                )
                return

            self.process_user_input(user_input)


def main():
    """Main entry point for the application"""
    # Create and run the app
    app = PDFQAApp()
    app.run()


if __name__ == "__main__":
    # Handle asyncio errors more gracefully
    import asyncio

    try:
        # For macOS - ensure a new event loop is created
        if sys.platform == "darwin":
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
