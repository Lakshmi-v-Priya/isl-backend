from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

import base64
import cv2
import numpy as np

from app.websocket_manager import ConnectionManager
from app.error_detection import is_blurry, is_low_light
from app.hand_detection import detect_hand
from app.landmark_extractor import extract_landmarks
from app.recognition import recognize_gesture
from app.sentence_builder import SentenceBuilder
from app.word_prediction import WordPredictor
from app.tts import text_to_speech

app = FastAPI()
manager = ConnectionManager()

# Sentence & prediction helpers
sentence_builder = SentenceBuilder()
predictor = WordPredictor()

# Serve audio files
app.mount("/audio", StaticFiles(directory="audio"), name="audio")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # ✅ RECEIVE JSON (IMPORTANT FIX)
            data = await websocket.receive_json()

            frame_data = data.get("frame")
            language = data.get("language", "English")

            if frame_data is None:
                await websocket.send_json({"message": "FRAME ERROR"})
                continue

            # ✅ Decode Base64 image
            try:
                header, encoded = frame_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"message": "FRAME DECODE ERROR"})
                continue

            if frame is None:
                await websocket.send_json({"message": "FRAME NULL"})
                continue

            # ---------- ERROR CHECKS ----------
            if is_low_light(frame):
                await websocket.send_json({"message": "ERROR: LOW LIGHT"})
                continue

            if is_blurry(frame):
                await websocket.send_json({"message": "ERROR: CAMERA BLURRY"})
                continue

            if not detect_hand(frame):
                await websocket.send_json({"message": "ERROR: NO HAND DETECTED"})
                continue

            # ---------- GESTURE RECOGNITION ----------
            landmarks = extract_landmarks(frame)
            if landmarks is None:
                await websocket.send_json({"message": "ERROR: LANDMARKS NOT FOUND"})
                continue

            word = recognize_gesture(landmarks)
            sentence_builder.add_word(word)
            sentence = sentence_builder.build_sentence()

            predictions = predictor.predict(word)

            # ---------- TEXT TO SPEECH ----------
            audio_path = text_to_speech(sentence, language)

            # ---------- SEND FINAL RESPONSE ----------
            await websocket.send_json({
                "sentence": sentence,
                "predictions": predictions,
                "audio": audio_path
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
