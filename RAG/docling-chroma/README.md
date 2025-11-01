# Document RAG System with Docling & ChromaDB

<img width="467" height="300" alt="image" src="https://github.com/user-attachments/assets/20933f0c-4cd4-4030-8d6a-183204eefcc7" />


A comprehensive Retrieval-Augmented Generation (RAG) system that processes documents using Docling and stores embeddings in ChromaDB for intelligent question-answering via AI agents.

## 📚 What is This Project?

This project provides a complete RAG implementation that:

- **Processes documents** using Docling locally (PDF, DOCX, PPTX, XLSX, HTML, TXT)
- **Generates embeddings** with ChromaDB's default embedding function (all-MiniLM-L6-v2)
- **Stores vectors** in ChromaDB for semantic search
- **Answers questions** via AI agents using [Chroma MCP server](https://github.com/chroma-core/chroma-mcp)
- **Strictly answers from documents** - No hallucinations or external information

**Key Features:**

- 🔧 **Local & Free** - Uses default local models (customizable to other models)
- 📊 **Structure Preservation** - Maintains document hierarchy and metadata
- 🎯 **Multiple Format Support** - Handles various document types seamlessly
- 🤖 **AI Agent Integration** - Query via MCP server with any AI agent
- 🚀 **Docker Integration** - Easy setup with Docker Compose
- 📝 **Interactive Notebooks** - Jupyter notebooks for experimentation

> **Note:** While this project uses local models by default (all-MiniLM-L6-v2 for embeddings, Docling for processing), you can configure it to use other embedding models or processing services as needed.

## 🎯 Project Structure

```
document-rag/
├── document-processing-local.ipynb          # ⭐ MAIN: Process documents locally with Docling
├── document-processing-remote-server.ipynb  # Alternative: Process via Docling server
├── docling-simple.ipynb                     # Demo: Simple Docling usage example
├── docker compose.yml                       # Docker services (Docling + ChromaDB)
├── pyproject.toml                           # Project dependencies
├── uv.lock                                  # Lock file for reproducible builds
├── chromadb/                                # ChromaDB persistent storage
└── .venv/                                   # Virtual environment (created by uv)
```

## 🛠️ Prerequisites

Before getting started, ensure you have:

- **Python 3.13+** installed
- **uv** package manager - [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker & Docker Compose** - [Installation guide](https://docs.docker.com/get-docker/)
- **Chroma MCP Server** - [Installation & setup](https://github.com/chroma-core/chroma-mcp)

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd document-rag
```

### Step 2: Install Dependencies with uv

```bash
# Install all project dependencies
uv sync

# This will:
# - Create a virtual environment in .venv/
# - Install all dependencies from pyproject.toml
# - Lock dependencies in uv.lock
```

### Step 3: Set Up Environment Variables (Optional)

If you want to disable ChromaDB telemetry:

```bash
# Optional: Customize ChromaDB telemetry
export ANONYMIZED_TELEMETRY=FALSE
```

**Note:** This project uses ChromaDB's default embedding function (all-MiniLM-L6-v2), so no API keys are required.

### Step 4: Start Docker Services

Start ChromaDB (and optionally Docling server):

```bash
docker compose up -d
```

This starts:

- **ChromaDB** on `http://localhost:5002` (required for RAG)
- **Docling Server** on `http://localhost:5001` (optional, for remote processing)

Verify services are running:

```bash
docker compose ps
```

You should see `chromadb` with status "Up" (and optionally `docling-serve`).

### Step 5: Configure Chroma MCP Server

To query your documents via AI agents, install and configure the [Chroma MCP server](https://github.com/chroma-core/chroma-mcp) in your AI client (Claude Desktop, Cline, etc.).

Follow the installation instructions in the [Chroma MCP repository](https://github.com/chroma-core/chroma-mcp).

## 📖 Usage Guide

### Starting Jupyter Notebooks with uv

To run any of the notebooks, use `uv run`:

```bash
# Start Jupyter Lab
uv run --with jupyter jupyter lab
```

This automatically uses the project's virtual environment without manual activation.

## 🔄 Complete Workflow: Document to Answer

### Step 1: Process and Ingest Documents (Local Processing - Recommended)

```bash
# Open the local processing notebook by running the following command:
uv run --with jupyter jupyter lab
```

**What it does:**

- Processes documents directly on your machine using Docling
- Fastest for most document sizes
- No network latency
- Full control over processing

**Processing Steps (in notebook):**

1. Set `input_file_path` (local file path) OR `input_url` (web URL)
2. Choose your collection name (e.g., "my-documents")
3. Run cells to:
   - Process document with Docling
   - Apply hybrid chunking with all-MiniLM-L6-v2 tokenizer (384 max tokens)
   - Generate embeddings using ChromaDB default embedding function
   - Store chunks in ChromaDB with metadata
4. Verify ingestion success

**Example:**

```python
# In document-processing-local.ipynb
input_file_path = "/path/to/your/document.pdf"
collection_name = "company-docs"

# Run notebook cells to process and ingest
```

### Step 2: Query Your Documents

You have two options for querying:

#### Option A: AI Agent with Chroma MCP (Recommended)

This is the **primary and recommended method** for querying your RAG system.

**Setup:**

1. Ensure [Chroma MCP server](https://github.com/chroma-core/chroma-mcp) is configured
2. Start a conversation with your AI agent
3. Provide this system prompt to your agent:

```
You are an AI agent that answers questions based on documents in a ChromaDB collection.

Rules:
- Only answer using information from the ChromaDB collection
- If you cannot find the answer, say "I don't know"
- Never guess or use external knowledge
- Always cite source documents

Collection name: [YOUR_COLLECTION_NAME]
```

**Example Interaction:**

```
You: "What are the two approaches to building custom AI agents?"

AI Agent: [Queries ChromaDB collection]
Based on the documents in your collection, the two approaches are:
1. Code-first approach - for maximum control
2. Application-first approach - for accelerated development

This information is from Chapter 7 of the startup_technical_guide_ai_agents_final.pdf document.
```

**Benefits:**

- ✅ Natural language querying
- ✅ Conversational follow-ups
- ✅ Automatic source attribution
- ✅ Context-aware answers
- ✅ No coding required

#### Option B: Chroma REST API

For programmatic access or custom integrations:

**Access the API documentation:**

```
http://localhost:5002/docs
```

**Example API Usage:**

```python
import requests

# Query the collection
response = requests.post(
    "http://localhost:5002/api/v1/collections/my-documents/query",
    json={
        "query_texts": ["What are the main findings?"],
        "n_results": 5
    }
)

results = response.json()
print(results)
```

**Use cases:**

- Custom applications
- Batch processing
- Integration with existing systems
- Automated workflows

## 🧪 Complete End-to-End Example

```bash
# 1. Start Docker services (ChromaDB)
docker compose up -d

# 2. Verify ChromaDB is running
docker compose ps

# 3. Start Jupyter for document processing
uv run jupyter notebook

# 4. In document-processing-local.ipynb:
#    - Set: input_file_path = "path/to/your/document.pdf"
#    - Set: collection_name = "my-docs"
#    - Run all cells to ingest document
#    - Verify: "Successfully added X chunks to ChromaDB"

# 5. Query via AI Agent (Recommended):
#    - Open your AI agent (Claude Desktop, Cline, etc.)
#    - Provide the system prompt (see above)
#    - Ask: "What are the main topics in the document?"
#    - Agent queries ChromaDB and provides answer with sources

# 6. Alternative - Query via API:
#    - Visit: http://localhost:5002/docs
#    - Use the interactive API documentation
#    - Test queries directly

# 7. When done:
docker compose down
```

## 📊 Supported Document Formats

Docling supports processing of:

| Format     | Extension | Notes                                 |
| ---------- | --------- | ------------------------------------- |
| PDF        | `.pdf`    | Complex layouts, tables, multi-column |
| Word       | `.docx`   | Full formatting preservation          |
| PowerPoint | `.pptx`   | Slide content extraction              |
| Excel      | `.xlsx`   | Table and data extraction             |
| HTML       | `.html`   | Web page processing                   |
| Text       | `.txt`    | Plain text documents                  |

## 📓 Notebook Descriptions

### 1. `document-processing-local.ipynb` ⭐ (MAIN WORKFLOW)

**Purpose:** Process and ingest documents into ChromaDB using local Docling processing.

**Use this for:**

- Primary document ingestion workflow
- Best performance for most use cases
- Direct control over processing
- Offline document processing

**Key steps:**

- Load document (file path or URL)
- Process with Docling locally
- Apply hybrid chunking
- Generate embeddings
- Store in ChromaDB

### 2. `document-processing-remote-server.ipynb` (ALTERNATIVE)

**Purpose:** Process documents via remote Docling server.

**Use this when:**

- Processing very large documents
- Sharing Docling resources across team
- Limited local compute resources
- Need Docling UI for debugging

**Key steps:**

- Send document to Docling server API
- Receive processed markdown
- Apply chunking and embedding locally
- Store in ChromaDB

### 3. `docling-simple.ipynb` (DEMO)

**Purpose:** Simple demonstration of basic Docling functionality.

**Use this for:**

- Learning Docling basics
- Quick testing of document processing
- Understanding Docling output format
- Experimentation

**Not part of main RAG workflow** - This is purely educational to show how Docling works with simple examples.

## 🔧 Configuration Options

### Default Embedding Configuration

This project uses **ChromaDB's default embedding function** (`sentence-transformers/all-MiniLM-L6-v2`):

- ✅ **No API keys required** - Runs locally
- ✅ **Fast & lightweight** - 384-dimensional embeddings
- ✅ **Zero cost** - Completely free
- ✅ **Customizable** - Can be replaced with other models

Example from the notebook:

```python
from docling.chunking import HybridChunker

MAX_TOKENS = 384  # Matches all-MiniLM-L6-v2 embedding dimension

chunker = HybridChunker(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)
```

To use a different embedding model, see [ChromaDB embedding functions documentation](https://docs.trychroma.com/embeddings).

### Adjusting Chunk Size

If you need different chunk sizes, modify the `MAX_TOKENS` parameter:

```python
from docling.chunking import HybridChunker

# Smaller chunks (better for precise retrieval)
MAX_TOKENS = 256

# Larger chunks (more context per chunk)
MAX_TOKENS = 512

chunker = HybridChunker(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=MAX_TOKENS,
    merge_peers=True
)
```

**Note:** Keep tokenizer consistent with your embedding model.

### ChromaDB Collection Settings

```python
# Create collection with custom metadata
collection = chroma_client.get_or_create_collection(
    name="my-documents",
    metadata={
        "description": "My custom document collection",
        "date_created": "2025-01-31"
    }
)
```

## 🐳 Docker Services Management

### Start Services

```bash
docker compose up -d
```

### Stop Services

```bash
docker compose down
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f chromadb
docker compose logs -f docling-serve
```

### Restart Services

```bash
docker compose restart
```

### Check ChromaDB Status

```bash
# Via curl
curl http://localhost:5002/api/v1/heartbeat

# Via browser
open http://localhost:5002/docs
```

## 📝 Dependencies

Key dependencies managed by `uv`:

- **chromadb** (1.3.0+) - Vector database with default embeddings (all-MiniLM-L6-v2)
- **docling** (2.59.0+) - Document processing library
- **pandas** (2.3.3+) - Data manipulation
- **ipywidgets** (8.1.7+) - Interactive notebook widgets
- **pydantic** (2.12.3+) - Data validation

## 🔍 Troubleshooting

### Issue: Jupyter kernel not found

**Solution:**

```bash
# Install ipykernel in the project environment
uv pip install ipykernel
uv run python -m ipykernel install --user --name=document-rag
```

### Issue: Docker services won't start

**Solution:**

```bash
# Check if ports are already in use
lsof -i :5002  # ChromaDB
lsof -i :5001  # Docling (optional)

# Stop conflicting services or change ports in docker compose.yml
```

### Issue: ChromaDB connection refused

**Solution:**

```bash
# Ensure ChromaDB is running
docker compose ps

# Restart ChromaDB
docker compose restart chromadb

# Check logs
docker compose logs chromadb

# Test connection
curl http://localhost:5002/api/v1/heartbeat
```

### Issue: Out of memory during local processing

**Solutions:**

1. **Process smaller documents** or split large PDFs
2. **Use remote server processing** (document-processing-remote-server.ipynb)
3. **Increase available RAM** or close other applications
4. **Adjust chunk size** to reduce memory footprint

### Issue: Embedding model download fails

**Solution:**

```bash
# ChromaDB will automatically download all-MiniLM-L6-v2 on first use
# If download fails, check internet connection or manually download:

# Install sentence-transformers
uv pip install sentence-transformers

# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue: MCP server not connecting

**Solution:**

1. Verify MCP configuration in your AI client
2. Restart your AI client after configuration changes
3. Check that ChromaDB is running: `docker compose ps`
4. Test ChromaDB API: `http://localhost:5002/docs`

### Issue: AI agent says "I don't know" for obvious content

**Possible causes:**

1. **Document not ingested** - Verify in notebook that chunks were added
2. **Wrong collection name** - Ensure agent queries correct collection
3. **Chunk size too small** - Content split across chunks, increase MAX_TOKENS to 512
4. **Query too specific** - Try broader questions

## 🎓 Learning Path

**Recommended progression:**

1. **Understand the basics** → `docling-simple.ipynb`

   - See how Docling processes documents
   - Understand output format
   - Experiment with different file types

2. **Process your first document** → `document-processing-local.ipynb`

   - Ingest a PDF or DOCX
   - Understand chunking strategy
   - See metadata preservation
   - Verify storage in ChromaDB

3. **Query via AI agent** → Configure MCP and interact

   - Set up Chroma MCP server
   - Provide system prompt
   - Ask questions about your document
   - Examine retrieved chunks and sources

4. **Explore API access** → `http://localhost:5002/docs`

   - Learn REST API endpoints
   - Test programmatic queries
   - Build custom integrations

5. **Optimize chunking** → Adjust MAX_TOKENS

   - Experiment with 256, 384, 512 tokens
   - Find balance between context and precision
   - Monitor retrieval quality

6. **Advanced usage** → Try remote processing
   - Compare local vs remote performance
   - Use Docling UI for debugging
   - Batch process multiple documents

## 💡 Key Concepts

### What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:

- **Retrieval**: Finding relevant document chunks using semantic search
- **Augmentation**: Adding retrieved context to prompts
- **Generation**: LLM generates answers grounded in retrieved content

**Benefits:**

- ✅ Reduces hallucinations
- ✅ Provides source attribution
- ✅ Updates knowledge without retraining
- ✅ Handles private/proprietary documents

### Why Docling?

Traditional text extraction fails on:

- Multi-column layouts
- Complex tables
- Images and diagrams
- Mixed content types

**Docling handles:**

- 🔧 OCR for scanned documents
- 📊 Table structure preservation
- 🎯 Layout analysis
- 🤖 Intelligent chunking
- 📝 Clean Markdown output

### Why ChromaDB?

**ChromaDB** is an open-source vector database that:

- Stores embeddings efficiently
- Provides fast similarity search
- Supports metadata filtering
- Runs locally or in production
- Integrates seamlessly with Python
- Offers REST API for any language

### Why MCP for Querying?

**Model Context Protocol (MCP)** enables:

- 🤖 Natural language interaction with your documents
- 💬 Conversational follow-up questions
- 🎯 Automatic relevance ranking
- 📚 Source attribution
- 🔄 Seamless AI agent integration

## 🔗 Additional Resources

### Documentation

- **Docling**: https://docling-project.github.io/docling/
- **ChromaDB**: https://docs.trychroma.com/
- **Chroma MCP Server**: https://github.com/chroma-core/chroma-mcp
- **all-MiniLM-L6-v2**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **uv Package Manager**: https://docs.astral.sh/uv/

### Related Projects

- **Docling GitHub**: https://github.com/DS4SD/docling
- **ChromaDB GitHub**: https://github.com/chroma-core/chroma
- **Model Context Protocol**: https://modelcontextprotocol.io/

## 🚀 Next Steps

Ready to customize this for your needs?

1. **Build your knowledge base** - Ingest multiple documents
2. **Tune chunk size** - Experiment with 256, 384, or 512 tokens for optimal retrieval
3. **Add metadata filtering** - Use metadata for scoped searches
4. **Create custom agents** - Build specialized AI agents for different document types
5. **Deploy to production** - Use cloud-hosted ChromaDB
6. **Monitor performance** - Track query accuracy and retrieval quality
7. **Scale embeddings** - Consider larger models for better accuracy if needed

## 💼 Production Considerations

When moving to production:

- **Security**: Implement authentication for ChromaDB
- **Scaling**: Consider cloud-hosted ChromaDB or self-hosted cluster
- **Monitoring**: Track query latency and result quality
- **Versioning**: Maintain collection versions for rollback
- **Backups**: Regular backup of ChromaDB data directory
- **Rate Limiting**: Protect API endpoints from abuse

## 📄 License

[Specify your license here]

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For questions or issues:

- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section above

---

Happy RAG building with AI agents! 🚀
