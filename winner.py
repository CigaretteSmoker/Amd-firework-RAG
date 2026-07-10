import streamlit as st
import json
import os
import time
import datetime
import hashlib
import io
import zipfile
import traceback
import base64
import re
from dotenv import load_dotenv
load_dotenv()  # Read local .env file if present
from pathlib import Path
from typing import List, Dict, Any, Generator
from openai import OpenAI, RateLimitError, APIConnectionError, BadRequestError

# --- RAG Specific Imports ---
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Parsers
try:
    from pypdf import PdfReader
except ImportError: PdfReader = None
try:
    from docx import Document
except ImportError: Document = None
import pandas as pd

# =====================================================================
# GLOBAL CONFIGURATIONS & CONSTANTS (256K CONTEXT OPTIMIZATION)
# =====================================================================
st.set_page_config(page_title="AMD Sovereign AI Brain (Fireworks AI)", page_icon="🧠", layout="wide")

CHROMA_BASE_PATH = "./chroma_db_amd"
Path(CHROMA_BASE_PATH).mkdir(parents=True, exist_ok=True)
SESSION_FILE = "agent_sessions_256k.json"

# --- RAG CONFIG FOR 256K CONTEXT ---
RAG_CONFIG = {
    "chunk_size": 1500,          # Large chunks for coherent context
    "chunk_overlap": 300,
    "top_k_retrieval": 15,       # Rich context retrieval
    "full_inject_threshold_tokens": 50_000, # If total docs < 50k tokens -> Inject full context
    "max_context_budget_tokens": 180_000,   # Safe budget for RAG + History
}

# --- SUPPORTED IMAGE FORMATS ---
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# =====================================================================
# TIME, TOKEN, AND FILE UTILITIES
# =====================================================================
def get_current_time_formatted() -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M:%S %p")

def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens (1 token ~ 4 characters)."""
    return max(1, len(text) // 4)

# =====================================================================
# ELITE MEDICAL PARSING & METADATA EXTRACTION (ZERO-LATENCY)
# =====================================================================
def extract_medical_profile(text: str) -> Dict[str, Any]:
    """Extracts critical patient metadata using zero-cost Regex for hackathon efficiency."""
    profile = {}
    
    # Extract Patient Name
    name_match = re.search(r"\*\*PATIENT NAME:\*\*\s*(.*)", text)
    if name_match: profile['patient_name'] = name_match.group(1).strip()
    
    # Extract Chief Allergy (CRITICAL FOR SAFETY)
    allergy_match = re.search(r"\*\*CHIEF ALLERGY:\*\*\s*(.*)", text)
    if allergy_match: profile['chief_allergy'] = allergy_match.group(1).strip()
    
    # Extract DOB / Age
    dob_match = re.search(r"\*\*DATE OF BIRTH:\*\*\s*(.*)", text)
    if dob_match: profile['dob_age'] = dob_match.group(1).strip()
    
    # Extract Medical Record Number
    rm_match = re.search(r"\*\*MEDICAL RECORD NUMBER:\*\*\s*(.*)", text)
    if rm_match: profile['mrn'] = rm_match.group(1).strip()
    
    return profile

def parse_medical_record_advanced(raw_text: str) -> tuple:
    """
    Parses structured medical text (Sovereign Medical Archive format) while preserving
    full SOAP context. Splits per SECTION, then per Visit Date within a section.

    Unlike a generic MarkdownHeaderTextSplitter (which can silently drop the
    'Section Title' metadata when a nested header like '**Date:' appears),
    this implementation guarantees that EVERY chunk keeps both:
      1. Its Section Title in metadata (critical for clinical traceability), and
      2. The original header lines inside page_content (nothing is stripped away).
    """
    section_re = re.compile(r"#### 📄 SECTION[^\n]*")
    date_re = re.compile(r"\*\*Date:\s*([^\n]*)")

    section_matches = list(section_re.finditer(raw_text))

    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    # Fallback: no recognizable "SECTION" headers found -> treat as one big chunk
    if not section_matches:
        return [raw_text.strip()], [{}]

    for idx, sec_match in enumerate(section_matches):
        sec_start = sec_match.start()
        sec_end = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(raw_text)
        section_text = raw_text[sec_start:sec_end]
        section_title = sec_match.group(0).strip()

        date_matches = list(date_re.finditer(section_text))

        if not date_matches:
            # Section with no explicit **Date:** field -> keep as single chunk
            chunks.append(section_text.strip())
            metadatas.append({"Section Title": section_title, "Visit Date": "N/A"})
            continue

        for d_idx, d_match in enumerate(date_matches):
            d_start = d_match.start()
            d_end = date_matches[d_idx + 1].start() if d_idx + 1 < len(date_matches) else len(section_text)
            sub_text = section_text[d_start:d_end].strip()
            visit_date = d_match.group(1).strip().rstrip("*").strip()

            # Keep the Section header line attached to the FIRST date-block of that section
            # so the clinical context ("SECTION I: EMERGENCY DEPARTMENT...") is never lost.
            if d_idx == 0:
                header_prefix = section_text[:d_start].strip()
                full_content = f"{header_prefix}\n{sub_text}" if header_prefix else sub_text
            else:
                full_content = f"{section_title}\n{sub_text}"

            chunks.append(full_content.strip())
            metadatas.append({"Section Title": section_title, "Visit Date": visit_date})

    return chunks, metadatas

# =====================================================================
# CHROMADB & UNIVERSAL EMBEDDING (OPENAI COMPATIBLE FOR ROCm / FIREWORKS)
# =====================================================================
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_BASE_PATH)

def get_active_embedding_function(fireworks_key: str):
    if not fireworks_key:
        return None
    # Fireworks AI exposes an OpenAI-compatible embeddings endpoint, so we reuse
    # ChromaDB's official OpenAIEmbeddingFunction (fully implements the required
    # EmbeddingFunction protocol: __call__, name(), get_config(), build_from_config()).
    return OpenAIEmbeddingFunction(
        api_key=fireworks_key,
        api_base="https://api.fireworks.ai/inference/v1",
        model_name="nomic-ai/nomic-embed-text-v1.5"
    )

def get_or_create_collection(session_id: str, emb_fn):
    client = get_chroma_client()
    safe_name = f"s_{session_id.replace('-', '').replace('.', '')}"
    return client.get_or_create_collection(name=safe_name, embedding_function=emb_fn)

# =====================================================================
# DOCUMENT PROCESSING (MULTIMODAL OCR VIA QWEN 3.7 VISION ON FIREWORKS)
# =====================================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CONFIG["chunk_size"],
    chunk_overlap=RAG_CONFIG["chunk_overlap"],
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def parse_image_with_fireworks(file_bytes: bytes, file_name: str, fireworks_api_key: str) -> str:
    """Extracts text (OCR) + visual descriptions from image using Qwen 3.7 Plus via Fireworks AI."""
    if not fireworks_api_key:
        raise RuntimeError("Fireworks API Key is required to process images/screenshots.")
    
    base64_image = base64.b64encode(file_bytes).decode("utf-8")
    ext = Path(file_name).suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}
    mime_type = f"image/{mime_map.get(ext, ext)}"
    
    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=fireworks_api_key
    )
    
    prompt = (
        "This image is a screenshot or document photo uploaded to the Knowledge Base. Please:\n"
        "1. Transcribe ALL visible text in this image verbatim (treat this as OCR).\n"
        "   If this is an error screenshot, code snippet, or a table, preserve the line structure and indentation as much as possible.\n"
        "2. Then, provide a concise description of the visual context (e.g., application type, UI design, product photo, diagram, chart, etc.).\n\n"
        "Output Format:\n=== EXTRACTED TEXT ===\n<content>\n\n=== VISUAL CONTEXT ===\n<content>"
    )
    
    resp = client.chat.completions.create(
        model="accounts/fireworks/models/qwen2-vl-72b-instruct", # Updated to stable vision model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )
    return resp.choices[0].message.content or ""

def parse_file(file_bytes: bytes, file_name: str, fireworks_api_key: str = None) -> str:
    ext = Path(file_name).suffix.lower()
    try:
        if ext == ".pdf":
            if not PdfReader: raise RuntimeError("Install 'pypdf'")
            reader = PdfReader(io.BytesIO(file_bytes))
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == ".docx":
            if not Document: raise RuntimeError("Install 'python-docx'")
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in [".csv", ".xlsx", ".xls"]:
            df = pd.read_csv(io.BytesIO(file_bytes)) if ext == ".csv" else pd.read_excel(io.BytesIO(file_bytes))
            return df.to_markdown(index=False)
        elif ext in IMAGE_EXTS:
            if fireworks_api_key:
                return parse_image_with_fireworks(file_bytes, file_name, fireworks_api_key)
            else:
                raise RuntimeError("Fireworks API Key is required to process images/screenshots.")
        else:
            return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Failed parsing '{file_name}': {e}")

def index_document(session_id: str, emb_fn, file_bytes: bytes, file_name: str, log_widget, fireworks_key: str = None) -> Dict[str, Any]:
    file_hash = get_file_hash(file_bytes)
    collection = get_or_create_collection(session_id, emb_fn)
    
    # Check duplication
    existing = collection.get(where={"file_hash": file_hash}, limit=1)
    if existing['ids']:
        return {"status": "skipped", "msg": f"⚠️ **'{file_name}'** is already indexed (skipped).", "chunks": 0, "tokens": 0, "medical_profile": None}

    ext_lower = Path(file_name).suffix.lower()
    tipe_label = "🖼️ Image/Screenshot (OCR via Fireworks Vision)" if ext_lower in IMAGE_EXTS else "📄 Document"
    log_widget.write(f"{tipe_label} **Processing:** `{file_name}` ({len(file_bytes)/1024:.1f} KB)...")
    
    raw_text = parse_file(file_bytes, file_name, fireworks_key)
    if not raw_text.strip():
        return {"status": "error", "msg": f"⚠️ **'{file_name}'** is empty.", "chunks": 0, "tokens": 0, "medical_profile": None}

    # --- ELITE MEDICAL DETECTION & PARSING ---
    is_medical_record = "**PATIENT NAME:**" in raw_text and "#### 📄 SECTION" in raw_text
    medical_profile = None
    
    if is_medical_record:
        log_widget.write("🩺 **Sovereign Medical Mode:** Detected structured clinical archive. Applying Markdown-Aware Chunking...")
        medical_profile = extract_medical_profile(raw_text)
        chunks, base_metadatas = parse_medical_record_advanced(raw_text)
        log_widget.write(f"✅ Preserved **{len(chunks)} clinical sections** without context fragmentation.")
    else:
        chunks = text_splitter.split_text(raw_text)
        base_metadatas = [{} for _ in chunks]
        log_widget.write(f"✂️ Split into **{len(chunks)} standard chunks**.")

    total_tokens = sum(estimate_tokens(c) for c in chunks)
    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    
    # Inject medical profile into EVERY chunk's metadata for safety enforcement
    metadatas = []
    for i in range(len(chunks)):
        meta = {
            "source": file_name, "file_hash": file_hash, "chunk_index": i,
            "indexed_at": datetime.datetime.now().isoformat(), "token_est": estimate_tokens(chunks[i])
        }
        meta.update(base_metadatas[i]) # Add markdown headers if medical
        if medical_profile:
            meta["patient_name"] = medical_profile.get("patient_name", "Unknown")
            meta["chief_allergy"] = medical_profile.get("chief_allergy", "None")
            meta["mrn"] = medical_profile.get("mrn", "Unknown")
        metadatas.append(meta)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        collection.add(documents=chunks[i:i+batch_size], metadatas=metadatas[i:i+batch_size], ids=ids[i:i+batch_size])
        
    return {
        "status": "success", 
        "msg": f"✅ **'{file_name}'** indexed successfully ({len(chunks)} chunks, ~{total_tokens:,} tok).", 
        "chunks": len(chunks), 
        "tokens": total_tokens,
        "medical_profile": medical_profile
    }

def retrieve_context(session_id: str, emb_fn, query: str, top_k: int) -> str:
    try:
        collection = get_or_create_collection(session_id, emb_fn)
        results = collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        if not docs: return "🔍 No relevant documents found."
        
        blocks = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            src = meta.get('source', 'Unknown')
            dist = meta.get('distance', 0)
            blocks.append(f"--- [Source {i+1}: {src} | Relevance: {1-dist:.2f}] ---\n{doc}\n")
        return "\n".join(blocks)
    except Exception as e:
        return f"❌ Retrieval Error: {e}"

def get_all_documents_context(session_id: str, emb_fn) -> str:
    """Retrieve all documents for the Full Injection Strategy."""
    try:
        collection = get_or_create_collection(session_id, emb_fn)
        all_data = collection.get(include=["documents", "metadatas"])
        docs = all_data['documents']
        metas = all_data['metadatas']
        if not docs: return ""
        
        file_map = {}
        for doc, meta in zip(docs, metas):
            src = meta.get('source', 'Unknown')
            file_map.setdefault(src, []).append(doc)
            
        blocks = []
        for src, chunks in file_map.items():
            blocks.append(f"=== FULL DOCUMENT: {src} ===\n" + "\n---\n".join(chunks) + "\n=== END OF DOCUMENT ===")
        return "\n\n".join(blocks)
    except Exception as e:
        return f"❌ Full Load Error: {e}"

def extract_zip_files(zip_bytes: bytes) -> List[tuple]:
    results = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            base_name = os.path.basename(info.filename)
            if not base_name or base_name.startswith(".") or "__MACOSX" in info.filename: continue
            try:
                data = z.read(info)
            except Exception:
                continue
            if data:
                results.append((info.filename, data))
    return results

def clear_knowledge_base(session_id: str):
    client = get_chroma_client()
    safe_name = f"s_{session_id.replace('-', '').replace('.', '')}"
    try:
        client.delete_collection(name=safe_name)
        return True
    except: return False

# =====================================================================
# SESSION PERSISTENCE (JSON)
# =====================================================================
def load_sessions() -> dict:
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return {}
    return {}

def save_sessions(sessions: dict):
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f: json.dump(sessions, f, ensure_ascii=False, indent=4)
    except Exception as e: st.error(f"Failed to save sessions: {e}")

# Init State
if "sessions" not in st.session_state:
    saved = load_sessions()
    if saved:
        st.session_state.sessions = saved
        st.session_state.current_session_id = list(saved.keys())[0]
    else:
        default_id = "session_default"
        st.session_state.sessions = {default_id: {"title": "Main Tab 🤖", "chat_history": [], "indexed_files": [], "total_indexed_tokens": 0, "medical_profile": None}}
        st.session_state.current_session_id = default_id

# =====================================================================
# TOOL IMPLEMENTATIONS (LASER-FOCUSED ON SECURE INTERNAL KNOWLEDGE RAG)
# =====================================================================
def rag_retrieve_tool(session_id: str, emb_fn, query: str, log_widget) -> str:
    log_widget.write(f"📚 RAG Retrieval (Top-{RAG_CONFIG['top_k_retrieval']}): `{query}`...")
    res = retrieve_context(session_id, emb_fn, query, RAG_CONFIG["top_k_retrieval"])
    log_widget.write("✅ Retrieval complete." if "No relevant documents" not in res else "⚠️ No results.")
    return res

# Tool Schema for Agentic Loop
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Retrieve specific contextual information from the Local Knowledge Base (user files). Use this tool to query deep metrics within the uploaded secure medical or legal archives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Specific search query for the medical or legal record database" }
                },
                "required": ["query"]
            }
        }
    }
]

def dispatch_tool(name: str, args: dict, log_widget, session_id, emb_fn):
    if name == "retrieve_knowledge": return rag_retrieve_tool(session_id, emb_fn, args["query"], log_widget)
    return f"Error: Tool '{name}' not recognized."

# =====================================================================
# SIDEBAR UI
# =====================================================================
with st.sidebar:
    st.header("⚙️ Fireworks AI Configuration")
    default_fw = st.session_state.get("fireworks_key") or os.getenv("FIREWORKS_API_KEY", "")
    fireworks_key = st.text_input("Fireworks API Key", type="password", value=default_fw, help="Get it from app.fireworks.ai")
    st.session_state.fireworks_key = fireworks_key

    MODEL_CHAIN = [
        "accounts/fireworks/models/minimax-m3",   # Primary: Ultra cost-effective ($0.15 / 1M tokens)
        "accounts/fireworks/models/gpt-oss-120b",  # Fallback 1: High intelligence ($0.90 / 1M tokens)
        "accounts/fireworks/models/mixtral-8x7b-instruct"     # Fallback 2: Fast & robust alternate
    ]
    api_url = "https://api.fireworks.ai/inference/v1"
    active_key = fireworks_key

    # Single Fireworks embedding function - same API key powers chat + embeddings
    emb_fn = get_active_embedding_function(fireworks_key=fireworks_key)

    st.markdown("---")
    st.header("🗂️ Tab Management")
    new_name = st.text_input("New Tab Name", placeholder="e.g., Clinical Cases Tab")
    if st.button("➕ Create Tab", use_container_width=True):
        if new_name.strip():
            nid = f"session_{int(time.time())}"
            st.session_state.sessions[nid] = {"title": new_name.strip(), "chat_history": [], "indexed_files": [], "total_indexed_tokens": 0, "medical_profile": None}
            st.session_state.current_session_id = nid
            save_sessions(st.session_state.sessions)
            st.rerun()

    opts = {sid: d["title"] for sid, d in st.session_state.sessions.items()}
    cur = st.session_state.current_session_id
    if cur not in opts: cur = list(opts.keys())[0]
    
    sel = st.selectbox("Active Tab:", list(opts.keys()), format_func=lambda x: opts[x], index=list(opts.keys()).index(cur))
    if sel != st.session_state.current_session_id:
        st.session_state.current_session_id = sel
        st.rerun()

    if st.button("🗑️ Delete Current Tab", type="primary", use_container_width=True):
        if len(st.session_state.sessions) > 1:
            clear_knowledge_base(st.session_state.current_session_id)
            del st.session_state.sessions[st.session_state.current_session_id]
            st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
            save_sessions(st.session_state.sessions)
            st.rerun()

    st.markdown("---")
    st.header("📚 Knowledge Base (RAG)")
    active_sess = st.session_state.sessions[st.session_state.current_session_id]
    
    st.caption(f"📊 **Est. Total Doc Tokens:** `{active_sess.get('total_indexed_tokens', 0):,}` tok")
    strategy = "🚀 **Full Injection** (All docs sent directly to LLM context)" if active_sess.get('total_indexed_tokens', 0) < RAG_CONFIG["full_inject_threshold_tokens"] else "🔍 **Top-K Retrieval** (ChromaDB Search)"
    st.caption(f"Active Strategy: {strategy}")

    # --- UI/UX: SOVEREIGN PATIENT DASHBOARD ---
    if active_sess.get("medical_profile"):
        prof = active_sess["medical_profile"]
        st.markdown("---")
        st.header("🩺 Sovereign Patient Profile")
        st.caption("Extracted via Zero-Latency Regex Engine")
        
        st.markdown(f"**Patient:** {prof.get('patient_name', 'Unknown')}")
        st.markdown(f"**MRN:** `{prof.get('mrn', 'Unknown')}` | **DOB:** {prof.get('dob_age', 'Unknown')}")
        
        # Critical Safety Alert Box
        if prof.get('chief_allergy') and 'none' not in prof['chief_allergy'].lower():
            st.error(f"⚠️ **CRITICAL ALLERGY:** {prof['chief_allergy']}")
            st.caption("This allergy is strictly enforced in the LLM System Prompt.")
        st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload Files / Screenshots (Choose multiple files)",
        accept_multiple_files=True,
        type=None,
        key=f"uploader_{st.session_state.current_session_id}"
    )
    
    if uploaded_files:
        if st.button("🚀 Index Files to KB", use_container_width=True):
            if emb_fn is None:
                st.error("Embedding initialization failed. Please make sure API keys/endpoints are properly set in the Sidebar.")
            else:
                with st.status("Processing & Embedding...", expanded=True) as status:
                    new_total_tokens = active_sess.get("total_indexed_tokens", 0)
                    for up_file in uploaded_files:
                        bytes_data = up_file.getvalue()
                        f_hash = get_file_hash(bytes_data)
                        if not any(f['hash'] == f_hash for f in active_sess.get("indexed_files", [])):
                            try:
                                res = index_document(
                                    st.session_state.current_session_id, 
                                    emb_fn, 
                                    bytes_data, 
                                    up_file.name, 
                                    status,
                                    fireworks_key=st.session_state.get("fireworks_key", "")
                                )
                                status.write(res["msg"])
                                if res["status"] == "success":
                                    active_sess.setdefault("indexed_files", []).append({"name": up_file.name, "hash": f_hash, "time": get_current_time_formatted()})
                                    new_total_tokens += res["tokens"]
                                    if res.get("medical_profile"):
                                        active_sess["medical_profile"] = res["medical_profile"]
                            except Exception as e:
                                status.write(f"⚠️ Skipped `{up_file.name}`: {e}")
                        else:
                            status.write(f"⏭️ `{up_file.name}` already exists.")
                    active_sess["total_indexed_tokens"] = new_total_tokens
                    save_sessions(st.session_state.sessions)
                    status.update(label="Indexing Complete!", state="complete")
                    st.rerun()

    st.markdown("")
    uploaded_zip = st.file_uploader(
        "Upload Folder (.zip)",
        type=["zip"],
        accept_multiple_files=False,
        key=f"zip_uploader_{st.session_state.current_session_id}"
    )
    
    if uploaded_zip:
        if st.button("📦 Extract & Index Folder", use_container_width=True):
            if emb_fn is None:
                st.error("Embedding initialization failed. Please make sure API keys/endpoints are properly set in the Sidebar.")
            else:
                with st.status("Extracting ZIP & Indexing...", expanded=True) as status:
                    try:
                        extracted = extract_zip_files(uploaded_zip.getvalue())
                    except Exception as e:
                        extracted = []
                        status.write(f"❌ Failed opening ZIP file: {e}")
                    status.write(f"📦 Found **{len(extracted)} files** inside folder.")
                    new_total_tokens = active_sess.get("total_indexed_tokens", 0)
                    for rel_path, bytes_data in extracted:
                        f_hash = get_file_hash(bytes_data)
                        if not any(f['hash'] == f_hash for f in active_sess.get("indexed_files", [])):
                            try:
                                res = index_document(
                                    st.session_state.current_session_id, 
                                    emb_fn, 
                                    bytes_data, 
                                    rel_path, 
                                    status,
                                    fireworks_key=st.session_state.get("fireworks_key", "")
                                )
                                status.write(res["msg"])
                                if res["status"] == "success":
                                    active_sess.setdefault("indexed_files", []).append({"name": rel_path, "hash": f_hash, "time": get_current_time_formatted()})
                                    new_total_tokens += res["tokens"]
                                    if res.get("medical_profile"):
                                        active_sess["medical_profile"] = res["medical_profile"]
                            except Exception as e:
                                status.write(f"⚠️ Skipped `{rel_path}`: {e}")
                        else:
                            status.write(f"⏭️ `{rel_path}` already exists.")
                    active_sess["total_indexed_tokens"] = new_total_tokens
                    save_sessions(st.session_state.sessions)
                    status.update(label="Folder Indexing Complete!", state="complete")
                    st.rerun()

    if active_sess.get("indexed_files"):
        st.write("**Indexed Files:**")
        for f in active_sess["indexed_files"]:
            st.caption(f"📄 {f['name']}  \n`{f['time']}`")

    if st.button("🧹 Clear KB of This Tab", use_container_width=True):
        if clear_knowledge_base(st.session_state.current_session_id):
            active_sess["indexed_files"] = []
            active_sess["total_indexed_tokens"] = 0
            active_sess["medical_profile"] = None
            save_sessions(st.session_state.sessions)
            st.success("KB cleared successfully."); st.rerun()

    st.markdown("---")
    st.header("📈 Benchmark (Telemetry)")
    if "last_token_usage" in st.session_state:
        u = st.session_state.last_token_usage
        st.metric("Prompt Tokens", f"{u.get('prompt', 0):,}")
        st.metric("Completion Tokens", f"{u.get('completion', 0):,}")
        st.metric("Total Tokens", f"{u.get('total', 0):,}")
        st.metric("Inference Speed", f"{u.get('speed', 0.0):.2f} tokens/s")
        st.metric("Latency (Response Time)", f"{u.get('latency', 0.0):.2f} s")
        cost = (u.get('prompt', 0) * 0.20 + u.get('completion', 0) * 0.60) / 1_000_000
        st.caption(f"Est. Cost (Cloud Equivalent): **${cost:.6f}**")

# =====================================================================
# MAIN CHAT AREA
# =====================================================================
active = st.session_state.sessions[st.session_state.current_session_id]
st.title(f"🧠 {active['title']}")
st.caption("Active Core: Fireworks AI (Serverless) | Context Window: 256K Ctx")

# Render History
for msg in active["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
user_q = st.chat_input("Ask anything about the secure medical/legal documents in this tab...")

if user_q:
    if not fireworks_key:
        st.warning("Please enter your Fireworks API Key in the Sidebar first!")
    elif emb_fn is None:
        st.warning("Please enter a valid Fireworks API Key so embeddings can be created!")
    else:
        active["chat_history"].append({"role": "user", "content": user_q})
        save_sessions(st.session_state.sessions)
        
        with st.chat_message("user"): st.markdown(user_q)
        
        # Client Setup - single Fireworks endpoint
        active_client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=fireworks_key)
        
        # --- SMART CONTEXT STRATEGY ---
        rag_context = ""
        total_doc_toks = active.get("total_indexed_tokens", 0)
        if total_doc_toks > 0:
            if total_doc_toks < RAG_CONFIG["full_inject_threshold_tokens"]:
                rag_context = get_all_documents_context(st.session_state.current_session_id, emb_fn)
                rag_note = f"[SYSTEM: FULL CONTEXT OF ALL DOCUMENTS ({total_doc_toks:,} tok) INJECTED DIRECTLY. ANSWER STRICTLY FROM THIS SOURCE.]"
            else:
                rag_context = ""
                rag_note = f"[SYSTEM: Total docs ({total_doc_toks:,} tok) > threshold. YOU MUST USE THE `retrieve_knowledge` TOOL TO SEARCH FOR CONTEXT.]"
        else:
            rag_note = "[SYSTEM: No documents in the KB for this tab.]"
        
        # --- SAFETY GUARDRAIL INJECTION ---
        med_profile = active.get("medical_profile") or {}
        allergy_warning = ""
        if med_profile.get("chief_allergy") and "none" not in med_profile["chief_allergy"].lower():
            allergy_warning = f"""
⚠️ CRITICAL SAFETY DIRECTIVE: 
The patient has a SEVERE, DOCUMENTED ALLERGY to: {med_profile['chief_allergy']}.
YOU MUST NEVER suggest, prescribe, recommend, or even mention this medication or its direct derivatives in your response. If a retrieved context mentions it, you must explicitly warn the user about the contraindication.
"""

        # --- SYSTEM PROMPT ELITE (FIREWORKS AI OPTIMIZED) ---
        sys_prompt = f"""You are the **AMD Sovereign AI Brain**, a highly secure, private RAG system running on the serverless Fireworks AI infrastructure.
Your context window supports up to 256,144 Tokens.
Your core directive is to analyze, summarize, and answer questions regarding highly sensitive medical, legal, or financial private records uploaded by the user with absolute data sovereignty.
{allergy_warning}
You have 1 External Tool:
`retrieve_knowledge`: Query local database archives in this specific tab.

📅 CURRENT DATE/TIME: {get_current_time_formatted()}

🧠 THINKING STRATEGY (Chain of Thought - Must Follow):
1. ANALYZE THE REQUEST: Pinpoint which user document or clinical record is being queried.
2. SELECT TOOL:
   - For User Files (PDF, DOCX, CSV, images, etc.) -> YOU MUST use `retrieve_knowledge` (if the KB is large) or READ FULL CONTEXT BELOW (if the KB is small).
3. MULTI-STEP RETRIEVAL: You can invoke `retrieve_knowledge` multiple times sequentially to pull isolated segments of a clinical history or legal contract together into a comprehensive analysis.
4. VERIFY: Ensure every claim is backed up verbatim by the source files (citation).
5. SYNTHESIS: Deliver a structured, concise, expert-level response in English.

📚 STATUS OF KNOWLEDGE BASE (THIS TAB):
{rag_note}
{rag_context}

⚠️ STRICT RULES:
- NEVER hallucinate or fabricate facts about the user's documents. If `retrieve_knowledge` returns empty -> state "Not found in the KB".
- Citations: Use the format `[Source X]` mapping to the label `[Source X: filename]` returned by the tool or context.
- Output: Standard English, highly professional, clinical, or formal legal tone, using clean structure (markdown, lists, headers, or tables)."""

        api_msgs = [{"role": "system", "content": sys_prompt}] + active["chat_history"]
        
        # --- AGENT LOOP WITH STREAMING STATUS ---
        with st.chat_message("assistant"):
            with st.status("🧠 Processing on AMD-Optimized Backend...", expanded=True) as status_box:
                final_answer = ""
                total_prompt_tokens = 0
                total_completion_tokens = 0
                run_latency = 0.0
                
                for step in range(12): # Max steps
                    status_box.write(f"🤔 **Step {step+1}:** Planning next steps...")
                    resp = None
                    last_err = None
                    
                    for model_name in MODEL_CHAIN:
                        try:
                            for attempt in range(3):
                                try:
                                    t_start = time.time()
                                    resp = active_client.chat.completions.create(
                                        model=model_name,
                                        messages=api_msgs,
                                        tools=TOOLS_SCHEMA,
                                        tool_choice="auto",
                                        temperature=0.2,
                                        max_tokens=8192,
                                    )
                                    t_end = time.time()
                                    run_latency += (t_end - t_start)
                                    break
                                except RateLimitError:
                                    if attempt < 2: time.sleep(2 ** attempt); continue
                                    raise
                                except APIConnectionError:
                                    if attempt < 2: time.sleep(2); continue
                                    raise
                            if resp is not None:
                                if model_name != MODEL_CHAIN[0]:
                                    status_box.write(f"⚠️ Primary model failed, failing over to: **{model_name}**")
                                break
                        except BadRequestError as e:
                            last_err = e
                            if "DEGRADED" in str(e).upper():
                                status_box.write(f"🔻 Model **{model_name}** status is DEGRADED, trying fallback model...")
                                continue
                            raise
                            
                    if resp is None:
                        raise last_err or RuntimeError("All models in MODEL_CHAIN failed to invoke.")
                    
                    msg = resp.choices[0].message
                    
                    # Accumulate Usage
                    if resp.usage:
                        total_prompt_tokens += resp.usage.prompt_tokens
                        total_completion_tokens += resp.usage.completion_tokens
                        status_box.write(f"📊 **Token Telemetry:** In: {resp.usage.prompt_tokens:,} | Out: {resp.usage.completion_tokens:,}")
                    
                    api_msgs.append(msg)
                    
                    if not msg.tool_calls:
                        final_answer = msg.content
                        status_box.update(label="✅ Reasoning Complete", state="complete", expanded=False)
                        break
                    
                    # Execute Tools
                    for tc in msg.tool_calls:
                        fname = tc.function.name
                        try:
                            fargs = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            fargs = {}; status_box.write(f"⚠️ Invalid JSON args: {tc.function.arguments}")
                        
                        status_box.write(f"🛠️ **Action:** `{fname}`(`{json.dumps(fargs, ensure_ascii=False)[:100]}...`)")
                        result = dispatch_tool(fname, fargs, status_box, st.session_state.current_session_id, emb_fn)
                        
                        if len(result) > 15000:
                            result = result[:15000] + "\n... [OUTPUT TRUNCATED FOR LENGTH] ..."
                            
                        api_msgs.append({
                            "tool_call_id": tc.id, "role": "tool", "name": fname, "content": result
                        })
                        
            # --- FINAL RENDER & TELEMETRY CALCULATION ---
            if final_answer:
                st.markdown(final_answer)
                active["chat_history"].append({"role": "assistant", "content": final_answer})
                save_sessions(st.session_state.sessions)
                
                tokens_sec = 0.0
                if total_completion_tokens > 0 and run_latency > 0:
                    tokens_sec = total_completion_tokens / run_latency
                    
                # Update Token Usage in Sidebar (Includes benchmark speeds)
                st.session_state.last_token_usage = {
                    "prompt": total_prompt_tokens,
                    "completion": total_completion_tokens,
                    "total": total_prompt_tokens + total_completion_tokens,
                    "speed": tokens_sec,
                    "latency": run_latency
                }
            else:
                st.error("Failed to generate final response.")