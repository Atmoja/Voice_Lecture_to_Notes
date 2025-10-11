import os
import tempfile
import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import whisper
model = whisper.load_model("base")
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Page Config & Styling
# -----------------------------
st.set_page_config(page_title="Lecture Assistant", page_icon="🎙️", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #243B55, #141E30); color: #f0f0f0; }
    h1, h2, h3 { color: #f8f9fa !important; }
    section[data-testid="stFileUploader"] {
        border: 2px dashed #9b59b6; padding: 20px; border-radius: 12px;
        background-color: rgba(255,255,255,0.1);
    }
    div.stButton > button {
        background: linear-gradient(90deg, #ff6f61, #d63384); color: white; border: none;
        border-radius: 12px; padding: 12px 20px; margin: 10px 0; font-size: 1.05em;
        font-weight: 600; cursor: pointer; box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease; width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #d63384, #ff6f61); transform: scale(1.03);
    }
    .download-btn > button {
        background: linear-gradient(90deg, #28a745, #218838) !important;
    }
    .download-btn > button:hover {
        background: linear-gradient(90deg, #218838, #28a745) !important;
    }
    .reset-btn > button {
        background: linear-gradient(90deg, #dc3545, #b02a37) !important;
    }
    .reset-btn > button:hover {
        background: linear-gradient(90deg, #b02a37, #dc3545) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# API Configuration
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Please set the GEMINI_API_KEY environment variable in Colab: `%env GEMINI_API_KEY=your_key_here`")
    st.stop()

genai.configure(api_key=api_key)
MODEL_NAME = "models/gemini-2.5-pro"
whisper_model = whisper.load_model("base")

# -----------------------------
# Helper Functions
# -----------------------------
def transcribe_audio(audio_file) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    result = whisper_model.transcribe(tmp_path)
    return result["text"]

def gemini_generate(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        output = model.generate_content(prompt)
        return getattr(output, "text", str(output))
    except Exception as e:
        return "⚠️ Gemini API error occurred. Please check your configuration."


def summarize_transcript(text: str) -> str:
    prompt = "Summarize the following lecture transcript into concise bullet points:\n\n" + text
    return gemini_generate(prompt)

def generate_quiz(text: str) -> list[dict]:
    prompt = (
        "Create 10 multiple-choice questions (4 options each). "
        "Mark the correct option with an asterisk (*). "
        "Format each as:\n"
        "Q: <question>\nA) <opt>\nB) <opt>\nC) <opt *>\nD) <opt>\nExplanation: <reason>\n\n"
        "Transcript:\n" + text
    )
    raw_output = gemini_generate(prompt)

    quiz = []
    q, opts, ans, exp = None, [], None, None
    for line in raw_output.splitlines():
        line = line.strip()
        if line.startswith("Q:"):
            if q:
                quiz.append({"question": q, "options": opts, "answer": ans, "explanation": exp})
            q, opts, ans, exp = line[2:].strip(), [], None, None
        elif line[:2] in ["A)", "B)", "C)", "D)"]:
            opt = line
            if "*" in opt:
                ans = opt.replace("*", "").strip()
            opts.append(opt.replace("*", "").strip())
        elif line.startswith("Explanation:"):
            exp = line.replace("Explanation:", "").strip()
    if q:
        quiz.append({"question": q, "options": opts, "answer": ans, "explanation": exp})
    return quiz

def generate_flashcards(text: str) -> list[tuple[str, str]]:
    prompt = (
        "Generate exactly 10 flashcards from this lecture transcript. "
        "Each must be formatted as:\nQ: <question>\nA: <answer>\n\nTranscript:\n" + text
    )
    raw_output = gemini_generate(prompt)
    flashcards, question, answer = [], None, None
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Q:"):
            if question and answer:
                flashcards.append((question, answer))
            question = line[2:].strip()
            answer = None
        elif line.startswith("A:"):
            answer = line[2:].strip()
            if question and answer:
                flashcards.append((question, answer))
                question, answer = None, None
    if question and answer:
        flashcards.append((question, answer))
    return flashcards[:10]

# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(summary=None, quiz=None, flashcards=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    if summary:
        content.append(Paragraph("<b>📝 Lecture Summary</b>", styles["Heading1"]))
        content.append(Spacer(1, 0.2 * inch))
        content.append(Paragraph(summary.replace("\n", "<br/>"), styles["Normal"]))
        content.append(Spacer(1, 0.4 * inch))

    if quiz:
        content.append(Paragraph("<b>📚 Quiz</b>", styles["Heading1"]))
        content.append(Spacer(1, 0.2 * inch))
        for i, q in enumerate(quiz, 1):
            content.append(Paragraph(f"<b>Q{i}. {q['question']}</b>", styles["Normal"]))
            for opt in q["options"]:
                content.append(Paragraph(f"- {opt}", styles["Normal"]))
            content.append(Paragraph(f"✅ Correct Answer: {q['answer']}", styles["Normal"]))
            if q["explanation"]:
                content.append(Paragraph(f"💡 {q['explanation']}", styles["Italic"]))
            content.append(Spacer(1, 0.2 * inch))

    if flashcards:
        content.append(Paragraph("<b>🔖 Flashcards</b>", styles["Heading1"]))
        content.append(Spacer(1, 0.2 * inch))
        for i, (q, a) in enumerate(flashcards, 1):
            content.append(Paragraph(f"<b>Q{i}:</b> {q}", styles["Normal"]))
            content.append(Paragraph(f"<b>A:</b> {a}", styles["Normal"]))
            content.append(Spacer(1, 0.15 * inch))

    doc.build(content)
    buffer.seek(0)
    return buffer

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎙️ Lecture Voice → Notes / Quiz / Flashcards")
st.write("Upload a lecture audio file. It will be transcribed and you can generate summary, quiz, or flashcards.")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded = st.file_uploader(
    "Upload lecture audio (wav, mp3, m4a)",
    type=["wav", "mp3", "m4a"],
    key=f"uploaded_file_{st.session_state.uploader_key}"
)

if uploaded is not None:
    if "transcript" not in st.session_state:
        with st.spinner("🔎 Transcribing audio..."):
            st.session_state.transcript = transcribe_audio(uploaded)

    st.subheader("📜 Transcript")
    st.write(st.session_state.transcript)

    # Initialize states
    for k in ["summary", "quiz", "flashcards"]:
        if k not in st.session_state:
            st.session_state[k] = None

    # Generate buttons
    if st.button("✨ Generate Summary"):
        with st.spinner("Generating summary..."):
            st.session_state.summary = summarize_transcript(st.session_state.transcript)

    if st.button("❓ Generate Quiz"):
        with st.spinner("Creating quiz..."):
            st.session_state.quiz = generate_quiz(st.session_state.transcript)

    if st.button("📑 Generate Flashcards"):
        with st.spinner("Creating flashcards..."):
            st.session_state.flashcards = generate_flashcards(st.session_state.transcript)

    # Show generated outputs
    if st.session_state.summary:
        st.subheader("📝 Summary")
        st.info(st.session_state.summary)

    if st.session_state.quiz:
        st.subheader("📚 Quiz")
        for i, q in enumerate(st.session_state.quiz, 1):
            st.markdown(f"**Q{i}. {q['question']}**")
            for opt in q["options"]:
                st.markdown(f"- {opt}")
            if st.button(f"Show Answer to Q{i}"):
                st.success(f"✅ Correct Answer: {q['answer']}")
                if q["explanation"]:
                    st.info(f"💡 {q['explanation']}")
            st.markdown("---")

    if st.session_state.flashcards:
        st.subheader("🔖 Flashcards")
        colors = ["#e3f2fd", "#fce4ec", "#e8f5e9", "#fff3e0", "#ede7f6",
                  "#f9fbe7", "#e0f7fa", "#f3e5f5", "#fbe9e7", "#e6ee9c"]
        total = len(st.session_state.flashcards)
        for row_start in range(0, total, 2):
            cols = st.columns(2)
            chunk = st.session_state.flashcards[row_start:row_start + 2]
            for local_idx, (col, (q, a)) in enumerate(zip(cols, chunk)):
                idx = row_start + local_idx + 1
                bg = colors[(idx - 1) % len(colors)]
                with col:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:12px; padding:16px;
                                    margin-bottom:14px; background-color:{bg};
                                    box-shadow:2px 2px 6px rgba(0,0,0,0.1);">
                            <p style="color:#000; font-weight:600;">Q{idx}: {q}</p>
                            <p style="color:#2c3e50;"><b>A:</b> {a}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # -----------------------------
    # FINAL SECTION (Vertical layout)
    # -----------------------------
    if st.session_state.summary and st.session_state.quiz and st.session_state.flashcards:
        st.markdown("---")
        st.markdown("### 🧾 Final Actions")

        pdf_buffer = generate_pdf(
            summary=st.session_state.summary,
            quiz=st.session_state.quiz,
            flashcards=st.session_state.flashcards
        )

        # Download Button (Green)
        with st.container():
            st.markdown('<div class="download-btn">', unsafe_allow_html=True)
            st.download_button(
                "📥 Download All as PDF",
                data=pdf_buffer,
                file_name="Lecture_Notes.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")  # vertical spacing

        # Reset Button (Red)
        with st.container():
            st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
            if st.button("🔄 Reset & Start Over", use_container_width=True):
                for key in ["transcript", "summary", "quiz", "flashcards"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.uploader_key += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


































