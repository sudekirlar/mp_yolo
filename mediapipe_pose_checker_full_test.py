# test_multipose_runner.py
import cv2
import json
import time
import threading
from typing import Optional, Tuple, List

from mediapipe_pose_checker import MultiPoseManager  # List döner.

# =========================
# 0) Konfig
# =========================
IS_VIDEO = True  # True: VIDEO_PATH, False: RTSP_URL
VIDEO_PATH = r"sample_videos/sample12.mp4"
RTSP_URL = r"rtsp://user:pass@192.168.1.10:554/stream1"

# Giriş çerçevesini döndürme konfigürasyonu
ROTATE_INPUT = False  # DÖNDÜRMEK İSTİYORSAN True bırak
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE  # 90° CCW

# Görüntü yüksekliği hedefi (0 demek ekranın %85'i demek.)
DISPLAY_HEIGHT = 0

PLAY_SPEED = 0.50          # 1.0 normal, 0.5 yavaş, 2.0 hızlı
DROP_LATE_FRAMES = False  # True ise geç kalmış kareleri atar (daha akıcı, ama atlama olabilir)
MAX_FRAME_DELAY_MS = 70   # Geç kalmış sayılacak eşik (drop için)

PRINT_JSON = False  # DEBUG İÇİNDİR!!!
SHOW_OVERLAY = False  # Overlay'i görmek için True yap ya da shortcut O'ya bas.
DRAW_BBOX = False  # Box görmek için True yap ya da shortcut B'ye bas.
WINDOW_NAME = "MultiPose Detection"


def get_screen_height() -> int:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        return int(user32.GetSystemMetrics(1))
    except Exception:
        return 1080


def resolve_display_height(target_h: int) -> int:
    if target_h and target_h > 0:
        return int(target_h)
    scr_h = get_screen_height()
    return max(480, int(scr_h * 0.85))


TARGET_DISPLAY_HEIGHT = resolve_display_height(DISPLAY_HEIGHT)


def resize_to_height(frame, target_h: int):
    if target_h <= 0:
        return frame
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, target_h), interpolation=interp)


def draw_multi_overlays(
    frame,
    results: List[Tuple[int, Optional[Tuple[int, int, int, int]], float]],
    fps: float,
    latency_ms: float
):
    shown = frame
    cls_map = {0: "T-POSE", 1: "ARMS-UP", -1: "None"}

    # BBox'lar (renk: 0=kırmızı, 1=turuncu, diğerleri=grims)
    if DRAW_BBOX:
        for idx, (_cid, bbox, _conf) in enumerate(results):
            if _cid == -1 or bbox is None:
                continue

            # OpenCV BGR
            if int(_cid) == 0:
                color = (0, 0, 255)       # Kırmızı
            elif int(_cid) == 1:
                color = (0, 165, 255)     # Turuncu
            else:
                color = (180, 180, 180)   # -1 için gri

            x, y, w, h = map(int, bbox)
            cv2.rectangle(shown, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            cv2.putText(
                shown,
                f"#{idx}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

    if SHOW_OVERLAY:
        lines = [f"Persons: {len(results)}", f"FPS: {fps:.1f}", f"Latency: {latency_ms:.1f} ms"]
        for idx, (cid, _bbox, conf) in enumerate(results):
            if cid == -1:
                continue
            lines.append(f"[{idx}] {cls_map.get(int(cid), str(cid))}  conf={conf:.2f}")

        x0, y0 = 12, 24
        for i, text in enumerate(lines):
            cv2.putText(shown, text, (x0, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return shown


class LatestFrameReader:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
        self._latest = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _worker(self):
        while not self._stopped.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            # --- ROTATE burada uygulanır (RTSP yolunda okuyucu thread içinde) ---
            if ROTATE_INPUT:
                try:
                    frame = cv2.rotate(frame, ROTATE_CODE)
                except Exception:
                    # rotation başarısız olsa da akış durmasın
                    pass
            with self._lock:
                self._latest = frame

    def read(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stopped.set()
        self._thread.join(timeout=1.0)


class VideoClock:
    def __init__(self, cap: cv2.VideoCapture, play_speed: float = 1.0):
        self.play_speed = max(0.05, float(play_speed))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.src_fps = fps if fps and fps > 1e-3 else 30.0
        self.frame_dt_ms = 1000.0 / self.src_fps
        self.t0 = time.perf_counter()
        self.base_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
        self.last_target_ms = self.base_ms

    def next_sleep(self, current_ms: Optional[float]) -> float:
        if current_ms is None or current_ms <= 0:
            target_ms = self.last_target_ms + self.frame_dt_ms
        else:
            target_ms = current_ms
        elapsed_wall = (time.perf_counter() - self.t0) * 1000.0
        ideal_wall_ms = (target_ms - self.base_ms) / self.play_speed
        sleep_ms = ideal_wall_ms - elapsed_wall
        self.last_target_ms = target_ms
        return sleep_ms / 1000.0


def build_capture() -> cv2.VideoCapture:
    if IS_VIDEO:
        # rotation parametresi **burada** verilemez; yalnızca backend seçilir.
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {VIDEO_PATH}")
        return cap
    else:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            raise RuntimeError(f"RTSP açılamadı: {RTSP_URL}")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap


def main():
    global SHOW_OVERLAY, DRAW_BBOX

    cap = build_capture()
    pipe = MultiPoseManager(topk=2)  # maksimum 2 kişi istiyoruz.

    reader = None
    if not IS_VIDEO:
        reader = LatestFrameReader(cap).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    frame_idx = 0
    t_prev = time.perf_counter()
    fps = 0.0
    fullscreen = False
    last_shown_shape = (0, 0)

    vclock = VideoClock(cap, PLAY_SPEED) if IS_VIDEO else None

    try:
        while True:
            t0 = time.perf_counter()

            if IS_VIDEO:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # --- ROTATE burada uygulanır (video dosyası yolunda) ---
                if ROTATE_INPUT:
                    try:
                        frame = cv2.rotate(frame, ROTATE_CODE)
                    except Exception:
                        pass
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            else:
                frame = reader.read()
                if frame is None:
                    time.sleep(0.002)
                    continue
                pos_ms = None  # RTSP için clock yok

            results = pipe.process(frame)  # List[(class_id, bbox, conf)] ŞEKLİNDE DÖNER!!!

            # DEBUG İÇİNDİR!!!
            if PRINT_JSON:
                out = {
                    "i": frame_idx,
                    "persons": [
                        {
                            "class": int(cid),
                            "bbox": list(map(int, bb)) if bb is not None else None,
                            "conf": float(cf),
                        }
                        for (cid, bb, cf) in results
                    ]
                }
                print(json.dumps(out, ensure_ascii=False))

            shown = frame
            if SHOW_OVERLAY or DRAW_BBOX:
                shown = frame.copy()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                shown = draw_multi_overlays(shown, results, fps=fps, latency_ms=latency_ms)

            shown = resize_to_height(shown, TARGET_DISPLAY_HEIGHT)

            h, w = shown.shape[:2]
            if last_shown_shape != (h, w) and not fullscreen:
                cv2.resizeWindow(WINDOW_NAME, w, h)
                last_shown_shape = (h, w)

            cv2.imshow(WINDOW_NAME, shown)

            t_now = time.perf_counter()
            dt = t_now - t_prev
            if dt > 0:
                instant = 1.0 / dt
                fps = fps * 0.9 + instant * 0.1
            t_prev = t_now
            frame_idx += 1

            if IS_VIDEO:
                sleep_s = vclock.next_sleep(pos_ms if pos_ms and pos_ms > 0 else None)
                if DROP_LATE_FRAMES and sleep_s < -MAX_FRAME_DELAY_MS / 1000.0:
                    # geç kalmış kareleri atla (oynatımı hızlandırmak istersen)
                    pass
                else:
                    if sleep_s > 0:
                        time.sleep(min(sleep_s, 0.2))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('o'):
                SHOW_OVERLAY = not SHOW_OVERLAY
                print(f"[INFO] SHOW_OVERLAY={SHOW_OVERLAY}")
            elif key == ord('b'):
                DRAW_BBOX = not DRAW_BBOX
                print(f"[INFO] DRAW_BBOX={DRAW_BBOX}")
            elif key == ord('f'):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)
                if not fullscreen:
                    last_shown_shape = (0, 0)

    finally:
        try:
            if reader is not None:
                reader.stop()
        except Exception:
            pass
        try:
            pipe.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
