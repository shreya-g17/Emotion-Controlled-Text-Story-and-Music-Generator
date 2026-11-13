import base64
import os
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from gtts import gTTS
from PIL import Image

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)

MUSIC_DIR = BASE_DIR / "music"
BANNER_PATH = BASE_DIR / "assets" / "moodmuse_bg.jpeg"

DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_NAME = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

# Tried in order after MODEL_NAME. All are text models with a free tier.
PREFERRED_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

# Models matching these words are NOT text generators (or have no free tier).
EXCLUDED_KEYWORDS = (
    "image", "imagen", "veo", "tts", "audio", "live", "embedding",
    "aqa", "robotics", "computer-use", "native-audio", "learnlm",
)

# On these errors we move on to the next model instead of failing.
SKIP_TO_NEXT_MODEL = {404, 429, 500, 502, 503, 504}
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

st.set_page_config(
    page_title="MoodMuse - Generative AI",
    page_icon="🎭",
    layout="wide",
)

# Each emotion has its own accent colour. The whole page picks it up
# (buttons, borders, glow), so the app "feels" like the selected mood.
EMOTIONS = {
    "Happy": {
        "emoji": "😊",
        "description": "Cheerful, optimistic and uplifting",
        "music_file": "happy.mp3",
        "accent": "#fbbf24",
        "image": BASE_DIR / "assets" / "emotions" / "happy.jpeg",
    },
    "Sad": {
        "emoji": "😢",
        "description": "Gentle, reflective and comforting",
        "music_file": "sad.mp3",
        "accent": "#60a5fa",
        "image": BASE_DIR / "assets" / "emotions" / "sad.jpeg",
    },
    "Angry": {
        "emoji": "😡",
        "description": "Calm, thoughtful and non-aggressive",
        "music_file": "angry.mp3",
        "accent": "#f87171",
        "image": BASE_DIR / "assets" / "emotions" / "angry.jpeg",
    },
    "Relaxed": {
        "emoji": "😌",
        "description": "Peaceful, slow and soothing",
        "music_file": "relaxed.mp3",
        "accent": "#34d399",
        "image": BASE_DIR / "assets" / "emotions" / "relaxed.jpeg",
    },
    "Fearful": {
        "emoji": "😨",
        "description": "Reassuring, gentle and encouraging",
        "music_file": "fearful.mp3",
        "accent": "#c084fc",
        "image": BASE_DIR / "assets" / "emotions" / "fearful.jpeg",
    },
}


# ---------------------------------------------------------------------------
# API CONFIGURATION
# ---------------------------------------------------------------------------

def get_ai_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Gemini API key not found. Check your .env file.")
    return genai.Client(api_key=api_key)


def is_text_model(name):
    lowered = name.lower()
    return not any(word in lowered for word in EXCLUDED_KEYWORDS)


def discover_text_models(client):
    """Ask the API for text-capable models, excluding image/audio/etc."""
    found = []
    for model in client.models.list():
        name = getattr(model, "name", None)
        actions = getattr(model, "supported_actions", None) or []
        if not name or "generateContent" not in actions:
            continue
        short = name.rsplit("/", 1)[-1]
        if is_text_model(short):
            found.append(short)
    # Prefer flash / lite models (cheaper, free-tier friendly)
    return sorted(
        set(found),
        key=lambda n: ("flash" not in n.lower(), n.lower()),
    )


def candidate_models(client):
    """Yield models lazily: configured one, known-good ones, then discovered."""
    seen = set()

    for name in [MODEL_NAME, *PREFERRED_FALLBACKS]:
        if name not in seen:
            seen.add(name)
            yield name

    try:
        discovered = discover_text_models(client)
    except errors.APIError:
        return

    for name in discovered[:5]:
        if name not in seen:
            seen.add(name)
            yield name


# ---------------------------------------------------------------------------
# PROMPT BUILDING
# ---------------------------------------------------------------------------

def build_prompt(emotion, content_type, topic, length):
    emotion_description = EMOTIONS[emotion]["description"]

    if content_type == "Story":
        task = f"""
        Write a completely original story of approximately {length} words.
        Include a beginning, middle and ending.
        Make the characters, events and setting creative.
        """
    elif content_type == "Text":
        task = f"""
        Write an original message of approximately {min(length, 200)} words.
        Make it meaningful, natural and emotionally expressive.
        Avoid generic quotes and copied phrases.
        """
    else:
        task = """
        Write original song lyrics.
        Include:
        1. Song title
        2. Verse 1
        3. Chorus
        4. Verse 2
        5. Musical style
        6. Recommended instruments and tempo

        Do not generate or claim to generate an audio file.
        """

    return f"""
    You are MoodMuse, an AI creative writing assistant.

    Generate completely new content based on the user's choices.

    Emotion: {emotion}
    Emotional guidance: {emotion_description}
    Content type: {content_type}
    Topic: {topic if topic.strip() else "Choose a creative topic"}
    Length: {length} words approximately.

    Instructions:
    - Create fresh content for every request.
    - Do not use predefined stories or messages.
    - Match the selected emotion throughout the content.
    - Follow the requested topic.
    - Use creative and natural language.
    - Do not explain your instructions.

    Task:
    {task}

    Return only the generated content.
    """


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

class EmptyResponseError(RuntimeError):
    pass


def call_model(client, model, prompt):
    """Call one model, retrying briefly on temporary server errors."""
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                raise EmptyResponseError(f"{model} returned an empty response.")
            return text
        except errors.APIError as error:
            if error.code in RETRYABLE_STATUS_CODES and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise


def generate_content(client, emotion, content_type, topic, length):
    prompt = build_prompt(emotion, content_type, topic, length)

    tried = []
    last_error = None

    for model in candidate_models(client):
        try:
            return call_model(client, model, prompt)
        except EmptyResponseError as error:
            last_error = error
            tried.append(f"{model} (empty response)")
        except errors.APIError as error:
            last_error = error
            tried.append(f"{model} (HTTP {error.code})")
            if error.code not in SKIP_TO_NEXT_MODEL:
                # Bad API key, invalid request, etc. Other models won't help.
                raise RuntimeError(
                    f"Gemini request failed for model '{model}' "
                    f"(HTTP {error.code}): {error}"
                ) from error

    summary = ", ".join(tried) or "none"
    raise RuntimeError(
        "No Gemini model could generate a response. "
        f"Models tried: {summary}. "
        "Check GEMINI_MODEL in your .env, your API key, and your quota."
    ) from last_error


# ---------------------------------------------------------------------------
# MEDIA HELPERS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_square_image(path_str):
    """Load an image and centre-crop it to a square. Cached for speed."""
    img = Image.open(path_str).convert("RGB")
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


@st.cache_data(show_spinner=False)
def load_banner_base64(path_str):
    with open(path_str, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_narration(text):
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.getvalue()


def show_music_player(emotion):
    st.markdown("#### 🎧 Listen to music")

    uploaded_audio = st.file_uploader(
        "Upload your own music (MP3 or WAV)",
        type=["mp3", "wav"],
        key=f"music_upload_{emotion}",
    )

    if uploaded_audio is not None:
        st.success("Music uploaded successfully!")
        st.audio(uploaded_audio.getvalue(), format=uploaded_audio.type)
        st.download_button(
            label="⬇️ Download uploaded music",
            data=uploaded_audio.getvalue(),
            file_name=uploaded_audio.name,
            mime=uploaded_audio.type,
        )
        return

    filename = EMOTIONS[emotion]["music_file"]
    music_path = MUSIC_DIR / filename

    if music_path.is_file():
        st.write("Or listen to the default track for this emotion:")
        st.audio(str(music_path), format="audio/mp3")
        st.caption(f"Track: {filename}")
    else:
        st.info(
            "No default music file found. "
            "Upload an MP3 or WAV file above to listen to music."
        )


# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------

# State is read first so the accent colour can be applied to the whole page.
if "selected_emotion" not in st.session_state:
    st.session_state.selected_emotion = "Happy"

emotion = st.session_state.selected_emotion
accent = EMOTIONS[emotion]["accent"]

# Per-emotion accent as CSS variables
st.markdown(
    f"""
    <style>
    :root {{
        --accent: {accent};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Main stylesheet (plain string, so CSS braces need no escaping)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=DM+Sans:wght@400;500;700&display=swap');

:root {
    --bg-1: #100d24;
    --bg-2: #1a1233;
    --surface: #211a38;
    --surface-2: #2b2147;
    --border: #3d3060;
    --text: #f5f0ff;
    --text-soft: #c9bde6;
}

html, body, .stApp, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
}

/* ---------- Page ---------- */
.stApp {
    background:
        radial-gradient(900px 500px at 85% -10%, color-mix(in srgb, var(--accent) 22%, transparent), transparent 70%),
        linear-gradient(160deg, var(--bg-1), var(--bg-2) 60%, #140f29);
    color: var(--text);
}

header[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* ---------- Text: every text element gets a readable colour ---------- */
h1, h2, h3, h4, h5 {
    font-family: 'Fraunces', Georgia, serif;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
p, li, label, span, .stMarkdown, [data-testid="stMarkdownContainer"] {
    color: var(--text);
}
[data-testid="stCaptionContainer"], .stCaption, small {
    color: var(--text-soft) !important;
}
hr { border-color: var(--border) !important; }

/* ---------- Hero banner ---------- */
.hero {
    text-align: center;
    padding: 44px 20px;
    border: 1px solid var(--accent);
    border-radius: 24px;
    margin-bottom: 28px;
    box-shadow: 0 0 40px color-mix(in srgb, var(--accent) 25%, transparent);
}
.hero h1 {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    color: #ffffff !important;
}
.hero p.tagline {
    font-size: 1.15rem;
    color: #f1e8ff !important;
    margin: 0;
}

/* ---------- Emotion info strip (replaces the invisible st.info) ---------- */
.mood-strip {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    margin: 14px 0 22px 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 5px solid var(--accent);
    border-radius: 14px;
}
.mood-strip .emoji { font-size: 1.8rem; }
.mood-strip .name  { font-weight: 700; color: var(--accent) !important; }
.mood-strip .desc  { color: var(--text) !important; }

/* ---------- Alerts (info / success / error / warning) ---------- */
div[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 14px;
}
div[data-testid="stAlert"] *,
div[data-testid="stAlert"] p {
    color: var(--text) !important;
}

/* ---------- Buttons ---------- */
.stButton > button,
.stDownloadButton > button {
    border-radius: 14px;
    padding: 0.6rem 1.2rem;
    font-weight: 700;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

/* Emotion cards (secondary) */
button[kind="secondary"],
button[data-testid="stBaseButton-secondary"] {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-2px);
}

/* Selected card + main action (primary) */
button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important;
    color: #1b1033 !important;
    border: none !important;
}
button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.1);
    transform: translateY(-2px);
    box-shadow: 0 8px 22px color-mix(in srgb, var(--accent) 40%, transparent);
}
button[kind="primary"] *,
button[data-testid="stBaseButton-primary"] * {
    color: #1b1033 !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--accent) !important;
}
.stDownloadButton > button * { color: var(--text) !important; }

/* ---------- Emotion images ---------- */
[data-testid="stImage"] img {
    border-radius: 18px;
    border: 2px solid var(--border);
}

/* ---------- Inputs ---------- */
.stTextInput input {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px;
}
.stTextInput input::placeholder { color: #9d8fc4 !important; }
.stTextInput input:focus { border-color: var(--accent) !important; }

/* Radio group */
div[role="radiogroup"] {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 12px 16px;
    border-radius: 14px;
}
div[role="radiogroup"] label p { color: var(--text) !important; }

/* Slider */
div[data-testid="stSlider"] * { color: var(--text) !important; }
div[data-testid="stSlider"] [role="slider"] { background-color: var(--accent) !important; }
div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] { color: var(--text-soft) !important; }

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--accent) !important;
    border-radius: 14px;
}
section[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* Expander + code block */
div[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
}
div[data-testid="stExpander"] * { color: var(--text) !important; }
pre, code { background: #0d0a1c !important; color: #f8e8ff !important; }

/* Result card (bordered container) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--accent) !important;
    border-radius: 20px;
    padding: 8px 10px;
}

/* ---------- Dark mode without config.toml ---------- */
:root { color-scheme: dark; }

/* Text input wrapper (the visible box around the input) */
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background-color: var(--surface-2) !important;
    border-radius: 12px;
}

/* Widget labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label { color: var(--text) !important; }

/* Spinner and status text */
div[data-testid="stSpinner"] * { color: var(--text) !important; }

/* Radio dots */
div[role="radiogroup"] label > div:first-child { border-color: var(--accent) !important; }

/* Audio player */
audio { width: 100%; border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------------------------

if BANNER_PATH.is_file():
    banner_css = (
        "linear-gradient(rgba(25,10,50,0.60), rgba(25,10,50,0.72)), "
        f"url('data:image/jpeg;base64,{load_banner_base64(str(BANNER_PATH))}')"
    )
else:
    banner_css = "linear-gradient(135deg, #2b1458, #4c1d95)"

st.markdown(
    f"""
    <div class="hero" style="background:{banner_css}; background-size:cover; background-position:center;">
        <h1>🎭 MoodMuse</h1>
        <p class="tagline">Turn your emotions into stories, words and music.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# EMOTION SELECTION CARDS
# ---------------------------------------------------------------------------

st.markdown("### ✨ How are you feeling today?")
st.caption("Pick a mood. The whole page will follow it.")

columns = st.columns(5, gap="medium")

for col, (name, info) in zip(columns, EMOTIONS.items()):
    with col:
        if info["image"].is_file():
            st.image(load_square_image(str(info["image"])), use_container_width=True)
        else:
            st.markdown(
                f"<div style='aspect-ratio:1;display:flex;align-items:center;"
                f"justify-content:center;font-size:3rem;background:#211a38;"
                f"border:2px solid #3d3060;border-radius:18px;'>{info['emoji']}</div>",
                unsafe_allow_html=True,
            )

        is_selected = st.session_state.selected_emotion == name
        if st.button(
            f"✓ {name}" if is_selected else name,
            key=f"emotion_card_{name}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_emotion = name
            st.rerun()

emotion = st.session_state.selected_emotion

st.markdown(
    f"""
    <div class="mood-strip">
        <span class="emoji">{EMOTIONS[emotion]['emoji']}</span>
        <div>
            <span class="name">{emotion}</span>
            <span class="desc"> — {EMOTIONS[emotion]['description']}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------

left, right = st.columns([1, 1], gap="large")

with left:
    content_type = st.radio(
        "What would you like the AI to create?",
        options=["Story", "Text", "Music"],
        horizontal=True,
    )

with right:
    topic = st.text_input(
        "Topic (optional)",
        placeholder="e.g. friendship, exams, adventure, nature",
    )

if content_type == "Story":
    length = st.slider(
        "Approximate story length (words)",
        min_value=100, max_value=500, value=200, step=50,
    )
elif content_type == "Text":
    length = st.slider(
        "Approximate message length (words)",
        min_value=30, max_value=200, value=80, step=10,
    )
else:
    length = 150

st.caption(f"Preferred AI model: {MODEL_NAME}")

if st.button("✨ Generate with AI", type="primary"):
    try:
        with st.spinner("The AI model is creating original content..."):
            client = get_ai_client()
            result = generate_content(
                client=client,
                emotion=emotion,
                content_type=content_type,
                topic=topic,
                length=length,
            )

        st.session_state["result"] = result
        st.session_state["result_emotion"] = emotion
        st.session_state["result_type"] = content_type
        st.session_state["generation_error"] = None
        st.session_state["narration_audio"] = None

    except Exception as error:
        st.session_state["result"] = None
        st.session_state["generation_error"] = f"{type(error).__name__}: {error}"

if st.session_state.get("generation_error"):
    st.error(
        "Content generation failed. Check your API key, "
        "internet connection, model availability, and API quota."
    )
    with st.expander("Technical details"):
        st.code(st.session_state["generation_error"])


# ---------------------------------------------------------------------------
# RESULT
# ---------------------------------------------------------------------------

if st.session_state.get("result"):
    result = st.session_state["result"]
    result_emotion = st.session_state["result_emotion"]
    result_type = st.session_state["result_type"]

    st.divider()

    with st.container(border=True):
        st.subheader(
            f"{EMOTIONS[result_emotion]['emoji']} Your {result_emotion.lower()} {result_type.lower()}"
        )
        st.markdown(result)

    if result_type == "Music":
        show_music_player(result_emotion)

    elif result_type in ["Story", "Text"]:
        if st.button("🔊 Generate narration"):
            try:
                with st.spinner("Generating audio narration..."):
                    audio_data = generate_narration(result)
                st.session_state["narration_audio"] = audio_data
            except Exception as error:
                st.error(f"Narration failed: {error}")

        if st.session_state.get("narration_audio"):
            st.markdown("#### 🎧 Listen to narration")
            st.audio(st.session_state["narration_audio"], format="audio/mp3")
            st.download_button(
                "⬇️ Download narration",
                data=st.session_state["narration_audio"],
                file_name="moodmuse_narration.mp3",
                mime="audio/mpeg",
            )

    st.download_button(
        label="⬇️ Download generated content",
        data=result,
        file_name=f"{result_emotion.lower()}_{result_type.lower()}.txt",
        mime="text/plain",
    )

st.divider()
st.caption("MoodMuse | Generative AI | Python, Streamlit and Gemini")