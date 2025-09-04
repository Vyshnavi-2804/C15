# app.py
"""
StudyMate — Enhanced Streamlit PDF Q&A app with Login
Features:
 - Secure login system
 - Upload PDFs; extract text (PyMuPDF), chunk, embed (SentenceTransformers), index (FAISS).
 - Ask natural-language questions; retrieve top chunks; generate grounded answers.
 - Providers: OpenAI (chat + image + whisper) OR Mock (extractive).
 - Text-to-speech (gTTS) for answers.
 - Audio transcription: upload an audio question (uses OpenAI Whisper if OPENAI_API_KEY provided).
 - Export / import FAISS index (.pkl)
 - Attractive UI layout and UX improvements.

Requirements:
 - Python 3.10+
 - pip install streamlit pymupdf sentence-transformers faiss-cpu openai gTTS python-dotenv
 - Set OPENAI_API_KEY env var if using OpenAI features (chat/image/whisper).
Run:
    streamlit run app.py
"""

from __future__ import annotations
import os
import io
import re
import json
import pickle
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_available = False
client = None

try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        openai_available = True
        print("✅ OpenAI API key loaded successfully!")
    else:
        print("❌ OpenAI API key NOT found in environment variables")
except ImportError:
    print("❌ OpenAI package not installed")
except Exception as e:
    print(f"❌ OpenAI initialization error: {e}")

# optional libraries
try:
    import faiss  # type: ignore
except Exception:
    faiss = None
    print("❌ FAISS not available")

try:
    from gtts import gTTS
except ImportError:
    print("❌ gTTS not installed")

# --------------------
# Login Module (Simplified - accepts any email/password)
# --------------------
class LoginModule:
    def __init__(self):
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "login_attempted" not in st.session_state:
            st.session_state.login_attempted = False

    def login_ui(self):
        st.markdown("""
            <style>
            .login-container {
                max-width: 400px;
                margin: 100px auto;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #1f77b4;'>🔑 StudyMate Login</h2>", unsafe_allow_html=True)

        email = st.text_input("📧 Email", placeholder="Enter your email")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        login_btn = st.button("🚀 Login", type="primary", use_container_width=True)

        if login_btn:
            st.session_state.login_attempted = True
            if email and password:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Please enter a valid email and password.")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
        st.caption("Any email and password accepted for login.")
        st.markdown("</div>", unsafe_allow_html=True)

    def is_logged_in(self):
        return st.session_state.logged_in

    def logout(self):
        st.session_state.logged_in = False
        st.session_state.login_attempted = False
        if "user_email" in st.session_state:
            del st.session_state.user_email
        st.rerun()

    def get_current_user(self):
        return st.session_state.get("user_email", "Unknown")

# --------------------
# Data structures
# --------------------
@dataclass
class ChunkMeta:
    file_name: str
    page_num: int  # 1-indexed
    chunk_id: int
    text: str
    start_char: int
    end_char: int

# --------------------
# Utilities
# --------------------
def clean_text(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def sentence_split(text: str) -> List[str]:
    pieces = re.split(r"(?<=[\.!?])\s+(?=[A-Z0-9\(\[\"'])", text)
    if len(pieces) == 1:
        pieces = re.split(r"\n+", text)
    return [p.strip() for p in pieces if p and not p.isspace()]

def chunk_sentences(sentences: List[str], target_chars: int = 1000, overlap_sentences: int = 1) -> List[str]:
    chunks, buf, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) + 1 > target_chars and buf:
            chunks.append(" ".join(buf).strip())
            buf = buf[-overlap_sentences:]
            cur_len = sum(len(x) for x in buf)
        buf.append(s)
        cur_len += len(s) + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks

def extract_pdf_chunks(file: io.BytesIO, file_name: str, target_chars: int = 1000) -> List[ChunkMeta]:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    metas: List[ChunkMeta] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        text = clean_text(page.get_text("text") or "")
        if not text:
            continue
        sents = sentence_split(text)
        chunks = chunk_sentences(sents, target_chars=target_chars)
        cursor = 0
        for idx, ch in enumerate(chunks):
            start = text.find(ch, cursor)
            end = start + len(ch) if start != -1 else cursor + len(ch)
            cursor = end
            metas.append(ChunkMeta(file_name, pno + 1, idx, ch, start if start != -1 else 0, end))
    return metas

# --------------------
# Embedding / FAISS Index
# --------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

class FAISSIndex:
    def __init__(self, dim: int):
        if faiss is None:
            raise RuntimeError("faiss-cpu is not installed. Please pip install faiss-cpu.")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.vectors = None  # numpy array float32
        self.metas: List[ChunkMeta] = []

    def add(self, vecs: np.ndarray, metas: List[ChunkMeta]):
        vecs = vecs.astype("float32")
        if self.vectors is None:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])
        self.metas.extend(metas)
        faiss.normalize_L2(self.vectors)
        self.index.reset()
        self.index.add(self.vectors)

    def search(self, qvec: np.ndarray, k: int = 5) -> List[Tuple[float, ChunkMeta]]:
        if self.vectors is None or len(self.metas) == 0:
            return []
        q = qvec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, min(k, len(self.metas)))
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            out.append((float(score), self.metas[int(idx)]))
        return out

    def dump(self) -> bytes:
        payload = {"metas": [m.__dict__ for m in self.metas], "vectors": self.vectors.tolist() if self.vectors is not None else [], "dim": self.dim}
        return pickle.dumps(payload)

    @staticmethod
    def load(blob: bytes) -> "FAISSIndex":
        payload = pickle.loads(blob)
        metas = [ChunkMeta(**d) for d in payload["metas"]]
        vectors = np.array(payload["vectors"], dtype="float32") if payload.get("vectors") else None
        dim = payload.get("dim", vectors.shape[1] if vectors is not None else 384)
        idx = FAISSIndex(dim)
        if vectors is not None and len(vectors):
            idx.add(vectors, metas)
        else:
            idx.metas = metas
        return idx

# --------------------
# Answer generation
# --------------------
def build_prompt(question: str, contexts: List[ChunkMeta]) -> str:
    header = (
        "You are StudyMate, a precise academic assistant. Use ONLY the provided context to answer and cite sources.\n"
        "- Cite sources inline as [filename pPAGE].\n"
        "- If answer is not in the provided materials say you don't know.\n"
        "- Be concise and structured.\n\n"
    )
    blocks = []
    for i, cm in enumerate(contexts, 1):
        blocks.append(f"<doc id=\"{i}\">{cm.text}\nSOURCE: [{cm.file_name} p{cm.page_num}]</doc>")
    context = "\n\n".join(blocks)
    return f"{header}Context:\n{context}\n\nQuestion: {question}\n\nAnswer (with citations):"

def generate_answer_extract(question: str, hits: List[ChunkMeta]) -> str:
    joined = "\n\n".join(h.text for h in hits)
    q_terms = [w for w in re.findall(r"\w+", question.lower()) if len(w) > 3]
    sents = sentence_split(joined)
    scored = [(sum(s.lower().count(t) for t in q_terms), s) for s in sents]
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    top = [s for sc, s in scored[:6] if sc > 0] or sents[:6]
    cites = sorted({f"[{h.file_name} p{h.page_num}]" for h in hits})
    answer = ("\n".join(top)).strip()
    return (answer + "\n\n" + " ".join(cites)) if answer else "I couldn't find an answer in the provided materials."

def generate_answer_openai_chat(question: str, hits: List[ChunkMeta]) -> str:
    if not openai_available or client is None:
        return "[OpenAI not available - check API key]"
    prompt = build_prompt(question, hits)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are StudyMate, an academic assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error: {e}]"

# --------------------
# OpenAI helper for images & transcription (if configured)
# --------------------
def openai_transcribe_audio(file_bytes: bytes, filename: str = "audio.wav") -> Tuple[bool, str]:
    """Return (success, transcription or error) - uses OpenAI whisper if available."""
    if not openai_available or client is None:
        return False, "OpenAI not available - check API key."
    try:
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".wav") as tf:
            tf.write(file_bytes)
            tf.flush()
            tfname = tf.name
        
        # Transcribe using OpenAI
        with open(tfname, "rb") as fh:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=fh
            )
        
        # Clean up temporary file
        os.unlink(tfname)
        
        return True, transcript.text
    except Exception as e:
        return False, f"Transcription error: {e}"

def openai_generate_image(prompt: str, n: int = 1, size: str = "1024x1024") -> Tuple[bool, List[str]]:
    """
    Returns (success, list_of_image_urls) or (False, error message)
    Uses OpenAI Images API.
    """
    if not openai_available or client is None:
        return False, ["OpenAI not available - check API key."]
    
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            n=n,
            quality="standard"
        )
        
        urls = [data.url for data in response.data]
        return True, urls
    except Exception as e:
        return False, [f"Image generation error: {e}"]

# --------------------
# Main App Function
# --------------------
def show_main_app(login_module):
    """Main application content after login"""
    
    # Custom CSS
    st.markdown(
        """
        <style>
          .stApp { background: linear-gradient(180deg,#f7f9fc,#ffffff); }
          .big-title { font-size:32px; font-weight:700; }
          .muted { color:#6b7280; }
          .card { background:rgba(255,255,255,0.9); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
          .user-info { background: #e8f5e8; padding: 8px 12px; border-radius: 8px; margin-bottom: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Header with user info
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.markdown(f"<div class='user-info'>👤 Logged in as: <strong>{login_module.get_current_user()}</strong></div>", unsafe_allow_html=True)
    with col3:
        if st.button("🚪 Logout"):
            login_module.logout()
    
    st.markdown("<div class='big-title' style='text-align: center;'>👋 Welcome to STUDYMATE.AI<br>📚 StudyMate — Conversational Q&A from your PDFs</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted' style='text-align: center;'>Upload PDFs, ask questions, get grounded answers with citations. Audio & image features available via OpenAI (optional).</div>", unsafe_allow_html=True)
    st.write("---")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙ Settings & Tools")
        provider = st.selectbox("LLM Provider", ["Mock (extractive)", "OpenAI Chat (if key set)"])
        embed_model = st.selectbox("Embedder", ["sentence-transformers/all-MiniLM-L6-v2"], index=0)
        chunk_chars = st.slider("Chunk size (characters)", 400, 2000, 1000, 100)
        top_k = st.slider("Retrieve top-K chunks", 1, 10, 5)
        st.markdown("---")
        st.subheader("Index Persistence")
        if st.button("💾 Export Index (downloadable)"):
            if "index" not in st.session_state or st.session_state.index is None:
                st.warning("No index to export.")
            else:
                blob = st.session_state.index.dump()
                st.download_button("Download index.pkl", data=blob, file_name="studymate_index.pkl", mime="application/octet-stream")
        import_blob = st.file_uploader("Import Index (.pkl)", type=["pkl"], accept_multiple_files=False)
        st.markdown("---")
        st.caption("OpenAI features (image/transcript) require OPENAI_API_KEY in environment.")
        
        # Display API key status
        if openai_available:
            st.success("✅ OpenAI API key is configured")
        else:
            st.error("❌ OpenAI API key not found. Some features will be disabled.")
        
        st.markdown("---")
        if st.button("🚪 Logout from Sidebar"):
            login_module.logout()

    # Initialize session state
    if "index" not in st.session_state:
        st.session_state.index = None
    if "embedder" not in st.session_state:
        try:
            st.session_state.embedder = get_embedder(embed_model)
        except Exception as e:
            st.session_state.embedder = None
            st.error(f"Error loading embedder: {e}")

    if import_blob is not None:
        try:
            st.session_state.index = FAISSIndex.load(import_blob.read())
            st.success("Imported index successfully.")
        except Exception as e:
            st.error(f"Failed to import index: {e}")

    # Layout columns
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("1) Upload PDFs & Build Index")
        uploaded = st.file_uploader("Upload one or more academic PDFs", type=["pdf"], accept_multiple_files=True, help="Drop PDFs here.")
        build = st.button("Build / Update Index", type="primary")
        st.markdown("*Indexed files:*")
        if st.session_state.index and len(getattr(st.session_state.index, "metas", [])):
            st.write(f"Chunks indexed: {len(st.session_state.index.metas)}")
        else:
            st.write("No index yet.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")  # spacer
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("2) Upload Audio Question🔊")
        audio_file = st.file_uploader("Upload audio (wav/mp3/m4a). I'll transcribe it.", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        transcribe_btn = st.button("Transcribe Audio")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")  # spacer
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("3) Image Generation📷")
        img_prompt = st.text_area("Image prompt", placeholder="e.g. diagram of a flood-rescue drone, schematic, or educational illustration")
        img_count = st.slider("Number of images", 1, 4, 1)
        img_size = st.selectbox("Size", ["1024x1024", "512x512"], index=0)
        generate_image = st.button("Generate Image(s)")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Ask a Question")
        question = st.text_input("Type your question here", placeholder="e.g., What is the main theorem and its assumptions?")
        ask_btn = st.button("Ask")
        st.markdown("</div>", unsafe_allow_html=True)

    # Build/update index logic
    if build and uploaded:
        try:
            with st.spinner("Extracting, chunking, embedding, and indexing..."):
                embedder = st.session_state.embedder or get_embedder(embed_model)
                all_metas: List[ChunkMeta] = []
                texts_for_embed = []
                for f in uploaded:
                    f_bytes = io.BytesIO(f.read())
                    metas = extract_pdf_chunks(f_bytes, f.name, target_chars=chunk_chars)
                    all_metas.extend(metas)
                    texts_for_embed.extend([m.text for m in metas])
                if not all_metas:
                    st.warning("No textual content found in uploaded PDFs.")
                else:
                    vecs = embedder.encode(texts_for_embed, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
                    dim = vecs.shape[1]
                    if st.session_state.index is None:
                        st.session_state.index = FAISSIndex(dim)
                    st.session_state.index.add(vecs, all_metas)
                    st.success(f"Indexed {len(all_metas)} chunks from {len(uploaded)} PDF(s).")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

    # Audio transcription
    transcription_text = ""
    if transcribe_btn:
        if audio_file is None:
            st.warning("Upload an audio file to transcribe.")
        else:
            audio_bytes = audio_file.read()
            # Try OpenAI transcription if configured
            if openai_available:
                with st.spinner("Transcribing with OpenAI Whisper..."):
                    ok, out = openai_transcribe_audio(audio_bytes, filename=audio_file.name)
                    if ok:
                        transcription_text = out
                        st.success("Transcription complete (OpenAI).")
                    else:
                        st.error(f"Transcription failed: {out}")
            else:
                # Fallback: ask user to paste text
                st.info("OpenAI key not configured — using fallback. Please paste the question or type it below.")
                transcription_text = st.text_area("Manual: paste or type your question", value="", height=120)

    # Allow transcription to pre-fill question
    if transcription_text:
        question = question or transcription_text
        st.info("Transcription placed into the question box. Edit if needed.")

    # Image generation
    if generate_image:
        if not img_prompt or img_prompt.strip() == "":
            st.warning("Please provide an image prompt.")
        elif not openai_available:
            st.error("Image generation requires the OpenAI API key (set OPENAI_API_KEY in env).")
        else:
            with st.spinner("Generating images..."):
                ok, result = openai_generate_image(img_prompt, n=img_count, size=img_size)
                if ok:
                    st.success("Image(s) generated.")
                    for i, url in enumerate(result, 1):
                        st.image(url, caption=f"Image {i}", use_column_width=True)
                else:
                    st.error(f"Image generation failed: {result}")

    # Ask / retrieve / answer flow
    if ask_btn:
        if not question or question.strip() == "":
            st.warning("Please enter a question (or transcribe/upload an audio question).")
        elif st.session_state.index is None:
            st.warning("Please upload PDFs and build the index first.")
        else:
            with st.spinner("Embedding question and retrieving context..."):
                embedder = st.session_state.embedder or get_embedder(embed_model)
                qvec = embedder.encode([question], convert_to_numpy=True)[0]
                results = st.session_state.index.search(qvec, k=top_k)
                hits = [m for _, m in results]

            if not hits:
                st.info("No relevant chunks found in index. Try increasing top-K or chunk size, or upload more PDFs.")
            else:
                with st.spinner("Generating answer..."):
                    answer = ""
                    if provider.startswith("OpenAI") and openai_available:
                        answer = generate_answer_openai_chat(question, hits)
                        if answer.startswith("[OpenAI error"):
                            st.info("OpenAI error — falling back to extractive.")
                            answer = generate_answer_extract(question, hits)
                    else:
                        # Mock extractive
                        answer = generate_answer_extract(question, hits)

                st.subheader("Answer")
                st.write(answer)

                # Text-to-speech
                try:
                    tts = gTTS(answer, lang="en")
                    tfile = "study_answer.mp3"
                    tts.save(tfile)
                    with open(tfile, "rb") as af:
                        st.audio(af.read(), format="audio/mp3")
                    st.caption("🔊 Audio version of the answer (gTTS)")
                except Exception as e:
                    st.error(f"Audio generation failed: {e}")

                # Show retrieved sources
                with st.expander("📄 Retrieved sources (top results)"):
                    for score, meta in results:
                        st.markdown(f"**{meta.file_name} — p{meta.page_num}** (score {score:.3f})")
                        st.write(meta.text)
                        st.markdown("---")

    st.markdown("---")
    st.caption("Built with Streamlit • PyMuPDF • SentenceTransformers • FAISS • OpenAI (optional). Use responsibly: answers are only as good as your uploaded materials.")

# --------------------
# Main Execution
# --------------------
def main():
    # Initialize login module
    login_module = LoginModule()
    
    # Check if user is logged in
    if not login_module.is_logged_in():
        login_module.login_ui()
        return  # Stop execution if not logged in
    
    # If logged in, show the main app
    show_main_app(login_module)

# Run the main function
if __name__ == "__main__":
    main()