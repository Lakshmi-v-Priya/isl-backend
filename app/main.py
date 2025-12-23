from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import base64, cv2, numpy as np

from app.websocket_manager import ConnectionManager
from app.hand_detection import detect_hand
from app.error_detection import is_blurry, is_low_light
from app.landmark_extractor import extract_landmarks
from app.recognition import recognize_gesture
from app.sentence_builder import SentenceBuilder
from app.word_prediction import WordPredictor
from app.tts import text_to_speech



app = FastAPI()
manager = ConnectionManager()

sentence_builder = SentenceBuilder()
predictor = WordPredictor()

# Serve audio files
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # 🔴 HERE: receive JSON from frontend
            data = await websocket.receive_json()

            image_data = data["frame"]
            language = data.get("language", "English")

            # Decode base64 image
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # --- ERROR CHECKS ---
            if is_low_light(frame):
                await websocket.send_json({"error": "LOW LIGHT"})
                continue

            if is_blurry(frame):
                await websocket.send_json({"error": "CAMERA BLURRY"})
                continue

            if not detect_hand(frame):
                await websocket.send_json({"error": "NO HAND DETECTED"})
                continue

            # --- RECOGNITION ---
            landmarks = extract_landmarks(frame)
            if landmarks is None:
                continue

            word = recognize_gesture(landmarks)

            sentence_builder.add_word(word)
            sentence = sentence_builder.build_sentence()

            predictions = predictor.predict(word)

            # 🔴 HERE: TEXT → SPEECH
            audio_path = text_to_speech(sentence, language)

            # 🔴 FINAL RESPONSE
            response = {
                "sentence": sentence,
                "predictions": predictions,
                "audio": audio_path
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
