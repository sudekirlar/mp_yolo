import cv2
import json
from mediapipe_pose_checker import MediapipePoseModule  # az önce koyduğun dosya adı

VIDEO_PATH = r"VID_20250827_164223.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {VIDEO_PATH}")

    pipe = MediapipePoseModule()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break  # video bitti

            class_id, bbox, conf = pipe.process(frame)

            # Terminale sade ve makinece okunur çıktı:
            # örn: {"i":123, "class":0, "bbox":[x,y,w,h], "conf":0.78}
            out = {
                "i": frame_idx,
                "class": int(class_id),
                "bbox": list(bbox) if bbox is not None else None,
                "conf": float(conf),
            }
            print(json.dumps(out, ensure_ascii=False))

            frame_idx += 1

            # İstersen kısmi hız sınırlama için uncomment:
            # cv2.waitKey(1)

    finally:
        try: pipe.close()
        except Exception: pass
        cap.release()

if __name__ == "__main__":
    main()

# T-POSE → 0, ARMS-UP → 1, None → -1