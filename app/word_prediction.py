from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ---- CORE MANAGERS ----
from app.websocket_manager import ConnectionManager
from app.sentence_builder import SentenceBuilder
from app.word_prediction import WordPredictor

# ---- VISION & ML ----
from app.hand_detection import detect_hand
from app.error_detection import is_blurry, is_low_light
from app.landmark_extractor import extract_landmarks
from app.recognition import recognize_gesture

# ---- UTILS ----
import base64
import cv2
import numpy as np


app = FastAPI()
manager = ConnectionManager()

# ---- NLP HELPERS ----
sentence_builder = SentenceBuilder()
predictor = WordPredictor()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            # ---------- DECODE BASE64 FRAME ----------
            header, encoded = data.split(",", 1)
            image_bytes = base64.b64decode(encoded)

            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                await manager.send_message("ERROR: INVALID FRAME", websocket)
                continue

            # ---------- ERROR CHECKS ----------
            if is_low_light(frame):
                await manager.send_message("ERROR: LOW LIGHT", websocket)
                continue

            if is_blurry(frame):
                await manager.send_message("ERROR: CAMERA BLURRY", websocket)
                continue

            # ---------- HAND DETECTION ----------
            if not detect_hand(frame):
                await manager.send_message("ERROR: NO HAND DETECTED", websocket)
                continue

            # ---------- LANDMARK EXTRACTION ----------
            landmarks = extract_landmarks(frame)

            if landmarks is None:
                await manager.send_message("ERROR: LANDMARK EXTRACTION FAILED", websocket)
                continue

            # ---------- WORD RECOGNITION ----------
            word = recognize_gesture(landmarks)

            if word is None:
                await manager.send_message("ERROR: GESTURE NOT RECOGNIZED", websocket)
                continue

            # ---------- SENTENCE BUILDING ----------
            sentence_builder.add_word(word)
            sentence = sentence_builder.build_sentence()

            # ---------- WORD PREDICTION ----------
            predictions = predictor.predict(word)

            # ---------- SEND RESPONSE ----------
            response = {
                "sentence": sentence,
                "predictions": predictions
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
