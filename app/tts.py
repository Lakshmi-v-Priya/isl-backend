from gtts import gTTS
import uuid
import os

# Folder to store generated audio files
AUDIO_DIR = "audio"

# Create audio folder if it does not exist
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

def cleanup_audio(folder=AUDIO_DIR, limit=20):
    """
    Keep only the latest 'limit' audio files
    to avoid storage overflow.
    """
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime
    )

    for file in files[:-limit]:
        try:
            os.remove(file)
        except:
            pass


def text_to_speech(text: str, language: str):
    """
    Converts text to speech in selected language
    and returns relative audio file path.
    """

    # Language mapping
    lang_map = {
        "English": "en",
        "Tamil": "ta",
        "Hindi": "hi",
        "Telugu": "te"
    }

    lang_code = lang_map.get(language, "en")

    # Generate unique filename
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    # Generate speech
    tts = gTTS(
        text=text,
        lang=lang_code,
        slow=False
    )
    tts.save(filepath)

    # Cleanup old audio files
    cleanup_audio()

    # Return path usable by frontend
    return f"audio/{filename}"
