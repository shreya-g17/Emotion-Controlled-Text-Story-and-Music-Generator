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

MODEL_NAME = (
    os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    or "gemini-3.8-flash"
)

st.set_page_config(
    page_title="MoodMuse - Generative AI",
    page_icon="🎭",
    layout="centered"
)

EMOTIONS = {
    "Happy": {
        "emoji": "😊",
        "description": "Cheerful, optimistic and uplifting",
        "music_file": "happy.mp3"
    },
    "Sad": {
        "emoji": "😢",
        "description": "Gentle, reflective and comforting",
        "music_file": "sad.mp3"
    },
    "Angry": {
        "emoji": "😡",
        "description": "Calm, thoughtful and non-aggressive",
        "music_file": "angry.mp3"
    },
    "Relaxed": {
        "emoji": "😌",
        "description": "Peaceful, slow and soothing",
        "music_file": "relaxed.mp3"
    },
    "Fearful": {
        "emoji": "😨",
        "description": "Reassuring, gentle and encouraging",
        "music_file": "fearful.mp3"
    }
}

# API CONFIGURATION

def get_ai_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Check your .env file."
        )

    return genai.Client(api_key=api_key)


def get_fallback_models(client, excluded_model):
    models = client.models.list()
    candidates = []

    for model in models:
        name = getattr(model, "name", None)
        supported_actions = getattr(model, "supported_actions", None) or []
        if (
            name
            and "generateContent" in supported_actions
            and name.rsplit("/", 1)[-1] != excluded_model
        ):
            candidates.append(name.rsplit("/", 1)[-1])

    return sorted(
        set(candidates),
        key=lambda name: ("flash" not in name.lower(), name.lower())
    )[:3]


# GENERATIVE AI CONTENT FUNCTION

def generate_content(client, emotion, content_type, topic, length):

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

    prompt = f"""
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

    retryable_status_codes = {500, 502, 503, 504}
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            content = getattr(response, "text", "")
            if not content:
                raise RuntimeError(
                    "Gemini returned an empty response. Try another prompt."
                )
            return content.strip()
        except errors.APIError as error:
            last_error = error
            if error.code in retryable_status_codes and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if error.code not in retryable_status_codes and error.code != 404:
                raise RuntimeError(
                    f"Gemini request failed for model '{MODEL_NAME}' "
                    f"(HTTP {error.code}): {error}"
                ) from error
            break

    try:
        fallback_models = get_fallback_models(client, MODEL_NAME)
    except errors.APIError as error:
        fallback_models = []
        fallback_discovery_error = error
    else:
        fallback_discovery_error = None

    for model_name in fallback_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            content = getattr(response, "text", "")
            if not content:
                raise RuntimeError(
                    "Gemini returned an empty response. Try another prompt."
                )
            return content.strip()
        except errors.APIError as error:
            if error.code not in retryable_status_codes and error.code != 404:
                raise RuntimeError(
                    f"Gemini request failed for fallback model "
                    f"'{model_name}' (HTTP {error.code}): {error}"
                ) from error
            last_error = error

    if last_error is None:
        raise RuntimeError("Gemini model could not generate a response.")

    if not fallback_models and last_error.code == 404:
        raise RuntimeError(
            f"Gemini model '{MODEL_NAME}' is unavailable and no alternate "
            "generation models were found for this API key. Check GEMINI_MODEL "
            "and the models enabled for your Gemini API key."
        ) from last_error

    fallback_note = (
        f" Model discovery also failed: {fallback_discovery_error}"
        if fallback_discovery_error
        else ""
    )
    tried_fallbacks = ", ".join(fallback_models) or "none"
    raise RuntimeError(
        f"Gemini could not generate content with model '{MODEL_NAME}' "
        f"(HTTP {last_error.code}); supported fallbacks tried: "
        f"{tried_fallbacks}. Try again later or check your API quota."
        f"{fallback_note}"
    ) from last_error

# MUSIC PLAYBACK

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


st.title("🎭 MoodMuse")
st.subheader(
    "Emotion-Controlled Story, Text and Music Generator"
)

st.write(
    "Select an emotion and let a Generative AI model create "
    "original content to match it."
)

st.divider()

# Emotion selection
emotion = st.selectbox(
    "Choose an emotion",
    options=list(EMOTIONS.keys()),
    format_func=lambda value: (
        f"{EMOTIONS[value]['emoji']} {value}"
    )
)

st.write(EMOTIONS[emotion]["description"])

# Content selection
content_type = st.radio(
    "What would you like the AI to create?",
    options=["Story", "Text", "Music"],
    horizontal=True
)

# User preferences
topic = st.text_input(
    "Enter a topic (optional)",
    placeholder="e.g. friendship, exams, adventure, nature"
)

if content_type == "Story":
    length = st.slider(
        "Approximate story length (words)",
        min_value=100,
        max_value=500,
        value=200,
        step=50
    )
elif content_type == "Text":
    length = st.slider(
        "Approximate message length (words)",
        min_value=30,
        max_value=200,
        value=80,
        step=10
    )
else:
    length = 150

st.caption(f"AI model: {MODEL_NAME}")

# Generate button
if st.button("✨ Generate with AI", type="primary"):
    try:
        with st.spinner(
            "The AI model is creating original content..."
        ):
            client = get_ai_client()

            result = generate_content(
                client=client,
                emotion=emotion,
                content_type=content_type,
                topic=topic,
                length=length
            )

        # Save the actual model response
        st.session_state["result"] = result
        st.session_state["result_emotion"] = emotion
        st.session_state["result_type"] = content_type
        st.session_state["generation_error"] = None

    except Exception as error:
        st.session_state["result"] = None
        st.session_state["generation_error"] = (
            f"{type(error).__name__}: {error}"
        )

# DISPLAY MODEL OUTPUT

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
        f"{EMOTIONS[result_emotion]['emoji']} "
        f"AI-Generated {result_type}"
    )

    st.markdown(result)

    if result_type == "Music":
        show_music_player(result_emotion)

    st.download_button(
        label="⬇️ Download generated content",
        data=result,
        file_name=(
            f"{result_emotion.lower()}_"
            f"{result_type.lower()}.txt"
        ),
        mime="text/plain"
    )

st.divider()
st.caption(
    "MoodMuse | Generative AI Project | Python, Streamlit and Gemini"
)
