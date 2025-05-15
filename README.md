# PDF Q&A with RAG - Chunker Comparison

A Streamlit application that demonstrates the difference between document chunking methods for Retrieval Augmented Generation (RAG) systems. This application allows you to ask questions about PDF documents and compares the results using two different chunking strategies:

1. **Docling's HybridChunker**: An intelligent chunking strategy that preserves semantic coherence
2. **RecursiveCharacterTextSplitter**: A standard text splitting approach based on character counts

## Features

- Interactive chat interface to ask questions about your PDF documents
- Side-by-side comparison of retrieved document chunks from different chunking methods
- Comparison of answers generated using both chunking methods
- Debugging tools to examine the full prompt and context sent to the language model
- Support for custom embedding models and language models

## Prerequisites

- Python 3.11 or higher
- One or more PDF documents for analysis

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/williamcaban/odsc-east-2025.git
cd odsc-east-2025
```

### 2. Install dependencies

If you don't have `uv` installed, you can install it first:

```bash
# Install uv (on macOS/Linux)
curl -sSf https://astral.sh/uv/install.sh | bash

# Install uv (on Windows PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

The project uses `uv` for fast dependency installation:

```bash
# Sync all dependencies from project.toml
uv sync
```

### 4. Create an environment file

Create a `.env` file in the project root with the following variables:
Note: Use `env.example` as template

```bash
# OpenAI API and compatible endpoints configuration

# OpenAI API credentials
OPENAI_API_KEY=your_key
OPENAI_MODEL_NAME=phi4:latest

# For custom OpenAI-compatible endpoints (like vLLM, local API servers, etc.)
# Comment out for official OpenAI API
# OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_BASE=http://localhost:11434/v1  # exmple using Ollama

# LLM parameters
LLM_TEMPERATURE=0.05
LLM_MAX_TOKENS=1024

# Embeddings configuration 
EMBED_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2 # default embedding model

# PDF configuration
PDF_FILE_PATH=./pdfs/2501.17887v1.pdf # default pdf
TOP_K=3  # Optional, number of chunks to retrieve
```

## Running the Application

### Start the Streamlit server:

```bash
uv run streamlit run streamlit_rag.py
```

The application will open in your default web browser at http://localhost:8501.

If you didn't specify a PDF file in the `.env` file, the application will search for any PDF in the current directory and its subdirectories.

## Using the Application

1. When the application loads, you'll see information about the loaded PDF in the sidebar
2. Use the chat input at the bottom to ask questions about the document
3. The system will display:
   - Retrieved document chunks from both chunking methods
   - An answer generated using the HybridChunker in the chat
   - A comparison of answers from both chunking methods below
   - Expandable debugging sections to examine the full context and prompt

## Troubleshooting

### Common Issues

- **Token length errors**: If you see token length warnings, the chunking parameters might need adjustment for your specific embedding model
- **MPS warnings on macOS**: These are informational and don't affect functionality
- **PDF loading errors**: Ensure your PDF is readable and not password-protected

### Debugging

The application includes debugging tools that help diagnose RAG pipeline issues:

1. Click on the "🔍 Debug: HybridChunker Full Prompt" or "🔍 Debug: RecursiveCharacterTextSplitter Full Prompt" expanders
2. Examine the retrieved documents and the full prompt sent to the model
3. Compare retrieval quality between the two chunking methods

## License

[Apache License 2.0](LICENSE)

```
Copyright 2025 ODSC East

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
