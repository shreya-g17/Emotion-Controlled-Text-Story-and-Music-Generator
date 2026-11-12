import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)

MUSIC_DIR = BASE_DIR / "music"

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
    layout="centered",
)

EMOTIONS = {
    "Happy": {
        "emoji": "😊",
        "description": "Cheerful, optimistic and uplifting",
        "music_file": "happy.mp3",
    },
    "Sad": {
        "emoji": "😢",
        "description": "Gentle, reflective and comforting",
        "music_file": "sad.mp3",
    },
    "Angry": {
        "emoji": "😡",
        "description": "Calm, thoughtful and non-aggressive",
        "music_file": "angry.mp3",
    },
    "Relaxed": {
        "emoji": "😌",
        "description": "Peaceful, slow and soothing",
        "music_file": "relaxed.mp3",
    },
    "Fearful": {
        "emoji": "😨",
        "description": "Reassuring, gentle and encouraging",
        "music_file": "fearful.mp3",
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
# MUSIC PLAYBACK
# ---------------------------------------------------------------------------

def show_music_player(emotion):
    filename = EMOTIONS[emotion]["music_file"]
    music_path = MUSIC_DIR / filename

    if music_path.is_file():
        st.subheader("🎧 Listen to matching music")
        st.audio(str(music_path), format="audio/mp3")
        st.caption(f"Track: {filename}")
    else:
        st.info(
            f"To enable audio playback, add a suitable MP3 file "
            f"named '{filename}' to the music folder. "
            "The AI-generated lyrics are available above."
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎭 MoodMuse")
st.subheader("Emotion-Controlled Story, Text and Music Generator")
st.write(
    "Select an emotion and let a Generative AI model create "
    "original content to match it."
)
st.divider()

emotion = st.selectbox(
    "Choose an emotion",
    options=list(EMOTIONS.keys()),
    format_func=lambda value: f"{EMOTIONS[value]['emoji']} {value}",
)
st.write(EMOTIONS[emotion]["description"])

content_type = st.radio(
    "What would you like the AI to create?",
    options=["Story", "Text", "Music"],
    horizontal=True,
)

topic = st.text_input(
    "Enter a topic (optional)",
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

if st.session_state.get("result"):
    result = st.session_state["result"]
    result_emotion = st.session_state["result_emotion"]
    result_type = st.session_state["result_type"]

    st.divider()
    st.subheader(
        f"{EMOTIONS[result_emotion]['emoji']} AI-Generated {result_type}"
    )
    st.markdown(result)

    if result_type == "Music":
        show_music_player(result_emotion)

    st.download_button(
        label="⬇️ Download generated content",
        data=result,
        file_name=f"{result_emotion.lower()}_{result_type.lower()}.txt",
        mime="text/plain",
    )

st.divider()
st.caption("MoodMuse | Generative AI Project | Python, Streamlit and Gemini")