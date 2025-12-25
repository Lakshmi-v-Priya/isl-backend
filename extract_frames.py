import cv2
import os

VIDEO_DIR = "datasets"
OUTPUT_DIR = "datasets"
FRAME_GAP = 1   # extract every frame

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in os.listdir(VIDEO_DIR):
    video_folder = os.path.join(VIDEO_DIR, label)
    image_folder = os.path.join(OUTPUT_DIR, label)

    os.makedirs(image_folder, exist_ok=True)

    for video_name in os.listdir(video_folder):
        if not video_name.lower().endswith(".mov"):
            continue

        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)

        count = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop center (VERY IMPORTANT for MediaPipe)
            h, w, _ = frame.shape
            frame = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

            img_path = os.path.join(
                image_folder,
                f"{video_name}_{saved}.jpg"
            )
            cv2.imwrite(img_path, frame)

            saved += 1
            count += 1

        cap.release()

print("✅ Frame extraction completed.")
