# test_multipose_runner.py

import cv2
import json
import time
import threading
from typing import Optional, Tuple, List

# Çoklu kişi modülü (liste döndürür)
from mediapipe_pose_checker import MultiPoseManager  # process(frame)->List[(class_id, bbox, conf)]

# =========================
# 0) Konfig
# =========================
IS_VIDEO = True  # True => VIDEO_PATH, False => RTSP_URL
VIDEO_PATH = r"sample_videos/sample7.mp4"
RTSP_URL = r"rtsp://user:pass@192.168.1.10:554/stream1"

# Görüntü yüksekliği hedefi (0 => ekran %85)
DISPLAY_HEIGHT = 0

# Oynatma hızı ve senkron
PLAY_SPEED = 1.0          # 1.0 normal, 0.5 yavaş, 2.0 hızlı
DROP_LATE_FRAMES = False  # True ise geç kalmış kareleri atar (daha akıcı, ama atlama olabilir)
MAX_FRAME_DELAY_MS = 50   # Geç kalmış sayılacak eşik (drop için)

PRINT_JSON = True
SHOW_OVERLAY = False
DRAW_BBOX = False
WINDOW_NAME = "Pipe Output"

# =========================
# 1) Yardımcılar
# =========================
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
    results: List[Tuple[int, Optional[Tuple[int,int,int,int]], float]],
    fps: float,
    latency_ms: float
):
    """
    Çoklu kişi için hafif overlay: BBox çizimi (opsiyonel) + üstte durum satırları.
    """
    shown = frame
    cls_map = {0: "T-POSE", 1: "ARMS-UP", -1: "None"}
    # BBox'lar
    if DRAW_BBOX:
        for idx, (_cid, bbox, _conf) in enumerate(results):
            # if _cid == -1 or bbox is None:  # <<< -1 olanları ve boş bbox'ları atla
            #     continue
            if bbox is None:  # <<< boş bbox'ları atla
                continue
            x, y, w, h = map(int, bbox)
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 190, 255), 2, cv2.LINE_AA)
            cv2.putText(shown, f"#{idx}", (x, max(0, y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,190,255), 1, cv2.LINE_AA)

    # Üst bilgilendirme
    if SHOW_OVERLAY:
        lines = [f"Persons: {len(results)}", f"FPS: {fps:.1f}", f"Latency: {latency_ms:.1f} ms"]
        # Kişi satırları
        for idx, (cid, _bbox, conf) in enumerate(results):
            # if cid == -1:
            #     continue  # MediaPipe poz bulamadı → atla
            lines.append(f"[{idx}] {cls_map.get(int(cid), str(cid))}  conf={conf:.2f}")

        x0, y0 = 12, 24
        for i, text in enumerate(lines):
            cv2.putText(shown, text, (x0, y0 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return shown

class LatestFrameReader:
    """RTSP için: arka planda sürekli en güncel frame'i tutar (buffer=1)."""
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
            with self._lock:
                self._latest = frame

    def read(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stopped.set()
        self._thread.join(timeout=1.0)

class VideoClock:
    """
    Video dosyası oynatma saatini gerçek zamanla eşler.
    - cap.get(POS_MSEC) varsa onu referans alır (VFR destekli).
    - Yoksa FPS'ten kare süresi hesaplar (CFR).
    - PLAY_SPEED ile hız ölçeklenir.
    """
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

# =========================
# 2) Ana Akış
# =========================
def main():
    global SHOW_OVERLAY, DRAW_BBOX

    cap = build_capture()
    pipe = MultiPoseManager(topk=2)  # maksimum 2 kişi

    # RTSP: arka plan okuyucu; Video: doğrudan read()
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

            # --- Frame al ---
            if IS_VIDEO:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break  # video bitti
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            else:
                frame = reader.read()
                if frame is None:
                    time.sleep(0.002)
                    continue
                pos_ms = None  # RTSP'de timestamp güvenilir değil

            # --- Pipe (çoklu kişi) ---
            results = pipe.process(frame)  # List[(class_id, bbox, conf)]

            # --- JSON çıktı ---
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

            # --- Görüntüle ---
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

            # --- FPS hesabı ---
            t_now = time.perf_counter()
            dt = t_now - t_prev
            if dt > 0:
                instant = 1.0 / dt
                fps = fps * 0.9 + instant * 0.1
            t_prev = t_now
            frame_idx += 1

            # --- Oynatma senkronu (yalnızca video dosyası) ---
            if IS_VIDEO:
                sleep_s = vclock.next_sleep(pos_ms if pos_ms and pos_ms > 0 else None)
                if DROP_LATE_FRAMES and sleep_s < -MAX_FRAME_DELAY_MS / 1000.0:
                    pass
                else:
                    if sleep_s > 0:
                        time.sleep(min(sleep_s, 0.2))

            # --- Tuşlar ---
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
