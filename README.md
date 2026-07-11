# AMD Sovereign AI Brain (Fireworks AI) 🧠

**AMD Sovereign AI Brain** is a highly secure, private Retrieval-Augmented Generation (RAG) system designed to analyze, summarize, and query sensitive medical, legal, or financial records. Leveraging the serverless infrastructure of Fireworks AI, this system ensures data sovereignty while providing an expansive context window for deep document analysis.

## 🚀 Key Features

- **Massive Context Window**: Optimized for up to **256,144 tokens**, allowing the system to process extremely long documents or large sets of files without losing coherence.
- **Multi-Format Document Support**:
    - **Text-based**: `PDF, DOCX, CSV`.
    - **Image-based**: `PNG, JPG, JPEG, WebP, BMP, GIF`.
- **Advanced Visual Intelligence**: Integrated OCR and visual description capabilities using **Qwen 3.7 Plus via Fireworks AI** to parse screenshots and document photos.
- **Hybrid Retrieval Strategy**:
    - **Top-K Retrieval**: Pulls the most relevant segments (Top 15) using ChromaDB for large knowledge bases.
    - **Full Context Injection**: Automatically injects the entire knowledge base into the prompt if the total token count is below 50,000, ensuring 100% accuracy for smaller datasets.
- **Secure Architecture**: Focused on data sovereignty, ensuring private records are handled with strict adherence to user-provided context.
- **Intelligent Chunking**: Uses `RecursiveCharacterTextSplitter` with optimized chunk sizes (1500 tokens) and overlap (300 tokens) for maximum semantic retention.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Infrastructure**: [Fireworks AI](https://fireworks.ai/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Orchestration**: LangChain (Text Splitters)
- **Parsing**: `pypdf`, `python-docx`, `pandas`
- **Embeddings**: OpenAI-compatible Embedding Function via Fireworks AI

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- A Fireworks AI API Key

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/CigaretteSmoker/Amd-firework-RAG.git
   cd Amd-firework-RAG
   '''
   
Install dependencies:

   ```bash
pip install -r requirements.txt
   ```
Environment Configuration: 

Create a .env file in the root directory and add your API key:
   ```env
FIREWORKS_API_KEY=your_api_key_here
   ``` 
Run the Application:

   ```
streamlit run winner.py
   ```
    
## 🧠 How It Works
The system follows a strict Chain of Thought (CoT) strategy to ensure professional-grade accuracy:

Analyze: Pinpoint which specific document or record is being queried.
Retrieve: Use the ``retrieve_knowledge`` tool to query the ChromaDB archive or read the full context if the dataset is small.
Verify: Ensure every claim is backed up verbatim by the source files with proper citations.
Synthesize: Deliver a structured, expert-level response.

<p align="center">
<img width="680" height="506" alt="architecture" src="https://github.com/user-attachments/assets/e6a06ace-510d-47bc-aaac-79e7ef0b41e0" />
</p>

## 📊 Configuration Details
The RAG engine is tuned for high-precision professional analysis:

## ⚙️ Configuration Parameters

Below are the optimized parameters used in this system to ensure high-quality retrieval and context management:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `chunk_size` | `1500` | Balanced for coherent context |
| `chunk_overlap` | `300` | Prevents loss of meaning between chunks |
| `top_k_retrieval` | `15` | Rich context retrieval for complex queries |
| `full_inject_threshold` | `50k tokens` | Switch to full-context mode for small KBs |
| `max_context_budget` | `180k tokens` | Reserved space for RAG + Chat History |

---
> Note: *Developed for secure, private, and sovereign AI intelligence.*
