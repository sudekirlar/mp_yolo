# ========= ENV: BLAS/OpenMP oversubscription engelle =========
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")

# ========= IMPORTS =========
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import deque
import math
import time
import threading
import queue
import random

# ========= 0) Konfig (Sadece pipeline’a gerekli parametreler) =========
MODEL_PATH = 'models/yolov8n.pt'
PERSON_CLASS_ID = 0
MODE = "YOLO_GUIDED_ONLY"                 # YOLO -> (stabil ROI) -> MediaPipe

# Eşikler ve parametreler
MIN_YOLO_CONF = 0.40
# Dinamik padding
BBOX_PADDING_BASE  = 36
BBOX_PADDING_TPOSE = 56
# MediaPipe
MP_MIN_DET = 0.60
MP_MIN_TRK = 0.50
MP_MODEL_COMPLEXITY = 1
# CLAHE tetikleyici
CLAHE_LUMA_THRESH = 50.0  # ROI ortalama gri < 50 → CLAHE

# Torch/cihaz
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

DEVICE = "cuda:0" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
HALF = DEVICE.startswith("cuda")

# Dinamik YOLO imgsz
YOLO_IMGSZ_BASE = 448
YOLO_IMGSZ_HIGH = 640
SHORT_SIDE_HIGH_IMGSZ_THRESH = 100  # <100 px ise bir sonraki detekte 640
MIN_SHORT_SIDE_MP = 64              # ROI kısa kenar <64 px ise MP atla

# ROI stabilizasyon
ROI_SMOOTH_ALPHA = 0.5
ROI_MAX_CENTER_JUMP_PCT = 0.035
ROI_MAX_SIZE_JUMP_PCT   = 0.04

# One-Euro
SMOOTH_IDX = [13, 14, 15, 16]
ONE_EURO_FREQUENCY   = 30.0
ONE_EURO_MINCUTOFF   = 0.8
ONE_EURO_BETA        = 0.015
ONE_EURO_DCUT        = 1.0

# Adaptif YOLO aralığı
DET_MIN, DET_MAX = 1, 8
LABEL_STABLE_RATIO = 0.70
UNKNOWN_STREAK_RESET = 3

# Pose eşikleri (geometri kanalı)
POSE_MIN_VIS           = 0.5
POSE_BUFFER_N          = 9
POSE_STABLE_MIN        = 4
TPOSE_SPINE_MIN, TPOSE_SPINE_MAX = 80.0, 105.0
TPOSE_ELBOW_MIN = 160.0
ARMSUP_SPINE_MAX = 35.0
ARMSUP_ELBOW_MIN = 130.0

# ========= 1) OpenCV ayarları =========
cv2.setUseOptimized(True)
try: cv2.setNumThreads(0)
except Exception: pass
try: cv2.ocl.setUseOpenCL(False)
except Exception: pass

# ========= 2) Yardımcılar =========
def clip_bbox(x, y, w, h, W, H) -> Tuple[int, int, int, int]:
    x = max(0, min(int(round(x)), W - 1))
    y = max(0, min(int(round(y)), H - 1))
    w = max(1, min(int(round(w)), W - x))
    h = max(1, min(int(round(h)), H - y))
    return x, y, w, h

def pad_bbox(x, y, w, h, pad, W, H) -> Tuple[int, int, int, int]:
    return clip_bbox(x - pad, y - pad, w + 2*pad, h + 2*pad, W, H)

def select_best_person_bbox(results, frame_shape, min_conf=MIN_YOLO_CONF):
    H, W = frame_shape[:2]
    best = None
    best_conf = 0.0
    best_score = -1.0
    for r in results:
        bxs = getattr(r, "boxes", None)
        if bxs is None:
            continue
        for box in bxs:
            cls = int(box.cls[0].item()) if box.cls is not None else -1
            if cls != PERSON_CLASS_ID:
                continue
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            if conf < min_conf:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(float(W-1), x2); y2 = min(float(H-1), y2)
            w = x2 - x1; h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            area = w * h
            score = conf * (1.0 + 0.0001 * area)
            if score > best_score:
                best_score = score
                best = (int(x1), int(y1), int(w), int(h))
                best_conf = conf
    return (best, best_conf) if best else (None, None)

def yolo_detect_person(model, frame, imgsz: int):
    try:
        if _HAS_TORCH:
            with torch.inference_mode():
                results = model(
                    frame, classes=[PERSON_CLASS_ID],
                    device=DEVICE, imgsz=imgsz, half=HALF, verbose=False
                )
        else:
            results = model(frame, classes=[PERSON_CLASS_ID], imgsz=imgsz, verbose=False)
    except TypeError:
        results = model(frame, classes=[PERSON_CLASS_ID], imgsz=imgsz, verbose=False)
    return select_best_person_bbox(results, frame.shape, MIN_YOLO_CONF)

def translate_landmarks_to_full(landmarks, roi_bbox, frame_shape):
    x, y, w, h = roi_bbox
    W_full, H_full = frame_shape[1], frame_shape[0]
    target = landmarks.pose_landmarks.landmark if hasattr(landmarks, "pose_landmarks") else landmarks.landmark
    invW, invH = 1.0 / W_full, 1.0 / H_full
    base_x, base_y = float(x), float(y)
    for lm in target:
        lm.x = (lm.x * w + base_x) * invW
    for lm in target:
        lm.y = (lm.y * h + base_y) * invH
    return landmarks

# ========= One-Euro ve eklem yumuşatma =========
class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    @staticmethod
    def alpha(cutoff, freq):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)
    def __call__(self, x, t=None):
        if t is None:
            t = time.perf_counter()
        if self.t_prev is None:
            self.t_prev = t; self.x_prev = x; self.dx_prev = 0.0
            return x
        dt = max(1e-6, t - self.t_prev)
        freq = 1.0 / dt
        dx = (x - self.x_prev) * freq
        a_d = self.alpha(self.dcutoff, freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff, freq)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat; self.dx_prev = dx_hat; self.t_prev = t
        return x_hat

class JointSmoother:
    def __init__(self, indices: List[int]):
        self.idx = indices
        self.fx: Dict[int, OneEuroFilter] = {i: OneEuroFilter(ONE_EURO_FREQUENCY, ONE_EURO_MINCUTOFF, ONE_EURO_BETA, ONE_EURO_DCUT) for i in indices}
        self.fy: Dict[int, OneEuroFilter] = {i: OneEuroFilter(ONE_EURO_FREQUENCY, ONE_EURO_MINCUTOFF, ONE_EURO_BETA, ONE_EURO_DCUT) for i in indices}
    def smooth(self, landmarks, frame_shape):
        W, H = frame_shape[1], frame_shape[0]
        lms = landmarks.landmark
        t = time.perf_counter()
        invW, invH = 1.0 / W, 1.0 / H
        for i in self.idx:
            if i >= len(lms):
                continue
            lm = lms[i]
            x_px, y_px = lm.x * W, lm.y * H
            lm.x = float(np.clip(self.fx[i](x_px, t) * invW, 0.0, 1.0))
            lm.y = float(np.clip(self.fy[i](y_px, t) * invH, 0.0, 1.0))
        return landmarks

# ========= ROI stabilizasyonu =========
class RoiStabilizer:
    def __init__(self, alpha=ROI_SMOOTH_ALPHA, max_c_jump_pct=ROI_MAX_CENTER_JUMP_PCT, max_s_jump_pct=ROI_MAX_SIZE_JUMP_PCT):
        self.alpha = alpha; self.max_c = max_c_jump_pct; self.max_s = max_s_jump_pct
        self.state = None
    def update(self, bbox: Tuple[int,int,int,int], frame_shape) -> Tuple[int,int,int,int]:
        W, H = frame_shape[1], frame_shape[0]
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        if self.state is None:
            self.state = clip_bbox(x, y, w, h, W, H); return self.state
        px, py, pw, ph = self.state
        pcx, pcy = px + pw/2, py + ph/2
        max_cx_jump = self.max_c * W; max_cy_jump = self.max_c * H
        max_w_jump  = self.max_s * W; max_h_jump  = self.max_s * H
        dcx = np.clip(cx - pcx, -max_cx_jump, max_cx_jump)
        dcy = np.clip(cy - pcy, -max_cy_jump, max_cy_jump)
        cx_c = pcx + dcx; cy_c = pcy + dcy
        dw = np.clip(w - pw, -max_w_jump, max_w_jump)
        dh = np.clip(h - ph, -max_h_jump, max_h_jump)
        w_c = max(1, pw + dw); h_c = max(1, ph + dh)
        sx = self.alpha * (cx_c - w_c/2) + (1.0 - self.alpha) * px
        sy = self.alpha * (cy_c - h_c/2) + (1.0 - self.alpha) * py
        sw = self.alpha * w_c + (1.0 - self.alpha) * pw
        sh = self.alpha * h_c + (1.0 - self.alpha) * ph
        self.state = clip_bbox(sx, sy, sw, sh, W, H); return self.state

# ========= 3) Pose sınıflandırma (2D sezgisel + geometri hibrit) =========
class PoseBuffer:
    def __init__(self, n=POSE_BUFFER_N):
        self.buf = deque(maxlen=n)
    def push(self, label: str):
        self.buf.append(label)
    def majority(self):
        if not self.buf:
            return "Unknown", 0, 0.0
        vals, cnts = np.unique(self.buf, return_counts=True)
        i = int(np.argmax(cnts))
        ratio = float(cnts[i]) / float(len(self.buf))
        return vals[i], int(cnts[i]), ratio

def _vec_angle_deg(v1, v2) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 90.0
    u1, u2 = v1 / n1, v2 / n2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))

def _elbow_angle_deg(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 180.0
    u1, u2 = v1 / n1, v2 / n2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(dot)))
    return ang if ang <= 180.0 else 360.0 - ang

def classify_pose_tpose_armsup(landmarks, frame_shape) -> Tuple[str, float, dict]:
    W, H = frame_shape[1], frame_shape[0]
    LM = landmarks.landmark
    def vis(i): return float(LM[i].visibility) if i < len(LM) else 0.0
    def get_px(i):
        if i >= len(LM): return np.array([np.nan, np.nan], dtype=np.float32)
        return np.array([LM[i].x * W, LM[i].y * H], dtype=np.float32)
    def avg_vis(idxs):
        vals = [vis(i) for i in idxs if i < len(LM)]
        return float(sum(vals)/len(vals)) if vals else 0.0

    core_idx = [11, 12, 23, 24]
    core_conf = avg_vis(core_idx)
    if core_conf < 0.60:
        return "Unknown", 0.0, {"reason":"low_core_visibility","core_conf":core_conf}

    LSh, RSh = get_px(11), get_px(12)
    LEl, REl = get_px(13), get_px(14)
    LWr, RWr = get_px(15), get_px(16)
    LHp, RHp = get_px(23), get_px(24)

    head_ids = [0,1,2]
    head_ys = [LM[i].y * H for i in head_ids if i < len(LM) and LM[i].visibility >= 0.3]
    head_y = float(np.mean(head_ys)) if head_ys else float(0.5 * (LSh[1]+RSh[1]) - 0.20*abs(LSh[1]-RSh[1]))

    mid_sh = 0.5*(LSh+RSh); mid_hp = 0.5*(LHp+RHp)
    spine_vec = (mid_sh - mid_hp)

    def arm_metrics(side: str):
        if side=='L': S,E,Wp = LSh,LEl,LWr; s,e,w = 11,13,15
        else:         S,E,Wp = RSh,REl,RWr; s,e,w = 12,14,16
        upper_vec = (E - S)
        spine_ang = _vec_angle_deg(spine_vec, upper_vec)
        v_s, v_e, v_w = vis(s), vis(e), vis(w)
        elbow_ang = None
        if v_e >= POSE_MIN_VIS and v_w >= 0.45:
            elbow_ang = _elbow_angle_deg(S, E, Wp)
        upper_vis = (v_s + v_e)/2.0
        return {"spine_ang":spine_ang,"elbow_ang":elbow_ang,"upper_vis":upper_vis,"wrist_vis":v_w}

    Lm, Rm = arm_metrics('L'), arm_metrics('R')

    def decide_arm(m):
        sa, ea = m["spine_ang"], m["elbow_ang"]
        t_upper_ok  = (TPOSE_SPINE_MIN < sa < TPOSE_SPINE_MAX)
        up_upper_ok = (sa < ARMSUP_SPINE_MAX)
        t_elbow_ok  = (ea is None) or (ea >= TPOSE_ELBOW_MIN)
        up_elbow_ok = (ea is None) or (ea >= ARMSUP_ELBOW_MIN)
        if t_upper_ok and t_elbow_ok: lbl="T"
        elif up_upper_ok and up_elbow_ok: lbl="UP"
        else: lbl="Unknown"
        if lbl=="T":
            orient_close=float(np.clip(1.0-abs(sa-92.5)/15.0,0.0,1.0))
            elbow_close =0.6 if (ea is None) else float(np.clip((ea-TPOSE_ELBOW_MIN)/20.0,0.0,1.0))
        elif lbl=="UP":
            up_close_fn=lambda x: float(np.clip(1.0 - x/(ARMSUP_SPINE_MAX+10.0),0.0,1.0))
            orient_close=up_close_fn(sa)
            elbow_close =0.6 if (ea is None) else float(np.clip((ea-ARMSUP_ELBOW_MIN)/30.0,0.0,1.0))
        else:
            orient_close=0.0; elbow_close=0.0
        geom_score=0.6*orient_close+0.4*elbow_close
        return lbl, geom_score

    L_lbl,L_geom = decide_arm(Lm)
    R_lbl,R_geom = decide_arm(Rm)

    partial_geom=False
    if L_lbl=="T" and R_lbl=="T":
        label_geom="T-POSE"; geom_pair=(L_geom+R_geom)/2.0
    elif L_lbl=="UP" and R_lbl=="UP":
        label_geom="ARMS-UP"; geom_pair=(L_geom+R_geom)/2.0
    else:
        if L_lbl in ("T","UP") and R_lbl=="Unknown":
            label_geom="T-POSE" if L_lbl=="T" else "ARMS-UP"; geom_pair=L_geom; partial_geom=True
        elif R_lbl in ("T","UP") and L_lbl=="Unknown":
            label_geom="T-POSE" if R_lbl=="T" else "ARMS-UP"; geom_pair=R_geom; partial_geom=True
        elif L_lbl in ("T","UP") and R_lbl in ("T","UP") and L_lbl!=R_lbl:
            if abs(L_geom-R_geom)>=0.25:
                winner=("L",L_lbl,L_geom) if L_geom>R_geom else ("R",R_lbl,R_geom)
                label_geom="T-POSE" if winner[1]=="T" else "ARMS-UP"; geom_pair=winner[2]; partial_geom=True
            else:
                label_geom="Unknown"; geom_pair=0.0
        else:
            label_geom="Unknown"; geom_pair=0.0

    def nx(p): return float(np.clip(p[0]/W,0.0,1.0))
    def ny(p): return float(np.clip(p[1]/H,0.0,1.0))
    spread_x_wrist    = abs(nx(LWr)-nx(RWr)) if not (np.isnan(LWr[0]) or np.isnan(RWr[0])) else 0.0
    spread_x_shoulder = abs(nx(LSh)-nx(RSh)) if not (np.isnan(LSh[0]) or np.isnan(RSh[0])) else 0.0
    y_wrist_mean      = (ny(LWr)+ny(RWr))/2.0 if not (np.isnan(LWr[1]) or np.isnan(RWr[1])) else ny(mid_sh)
    y_shoulder_mean   = ny(mid_sh)
    y_head_norm       = head_y / H

    t_spread_score   = float(np.clip((spread_x_wrist-0.30)/0.20,0.0,1.0))
    t_y_align_score  = float(np.clip(1.0 - abs(y_wrist_mean - y_shoulder_mean)/0.07,0.0,1.0))
    t2d = 0.7*t_spread_score + 0.3*t_y_align_score

    up_vertical_score     = float(np.clip(((y_head_norm - y_wrist_mean)-0.03)/0.10,0.0,1.0))
    up_spread_small_score = float(np.clip(1.0 - (spread_x_wrist - 1.1*max(1e-6,spread_x_shoulder))/0.15,0.0,1.0))
    up2d = 0.7*up_vertical_score + 0.3*up_spread_small_score

    if (t2d-up2d)>0.10 and t2d>=0.55:
        label_2d="T-POSE"; two_d_score=t2d
    elif (up2d-t2d)>0.10 and up2d>=0.55:
        label_2d="ARMS-UP"; two_d_score=up2d
    else:
        label_2d="Unknown"; two_d_score=max(t2d,up2d)

    label = label_2d if label_2d!="Unknown" else label_geom

    upper_vis_avg  = (Lm["upper_vis"] + Rm["upper_vis"])/2.0
    wrists_vis_avg = (Lm["wrist_vis"]+ Rm["wrist_vis"])/2.0
    conf_visibility = float(np.clip(0.80*upper_vis_avg + 0.20*wrists_vis_avg, 0.0, 1.0))

    conf = 0.70*float(np.clip(two_d_score,0.0,1.0)) + \
           0.30*float(np.clip(0.80*conf_visibility + 0.20*float(np.clip(geom_pair,0.0,1.0)),0.0,1.0))
    if label_geom in ("T-POSE","ARMS-UP") and partial_geom:
        conf *= 0.92
    if label in ("T-POSE","ARMS-UP") and upper_vis_avg>=0.60:
        conf = max(conf, 0.60)
    conf = float(np.clip(conf,0.0,1.0))

    dbg = {"core_conf":core_conf,"upper_vis_avg":upper_vis_avg,"wrists_vis_avg":wrists_vis_avg,
           "geom_pair":float(np.clip(geom_pair,0.0,1.0)),
           "two_d":{"t2d":t2d,"up2d":up2d,"spread_x_wrist":spread_x_wrist,"spread_x_shoulder":spread_x_shoulder,
                    "y_wrist_mean":y_wrist_mean,"y_shoulder_mean":y_shoulder_mean,"y_head_norm":y_head_norm},
           "labels":{"geom":label_geom,"two_d":label_2d,"final":label}}
    return label, conf, dbg

# ========= 4) MediaPipe Worker =========
mp_pose = mp.solutions.pose

class PoseWorker(threading.Thread):
    def __init__(self):
        super().__init__(name="PoseWorker", daemon=True)
        self.in_q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        self.out_lock = threading.Lock()
        self.last_label = "Unknown"
        self.last_conf  = 0.0
        self.last_time  = 0.0
        self.last_ok    = False
        self.pose = mp_pose.Pose(
            min_detection_confidence=MP_MIN_DET,
            min_tracking_confidence=MP_MIN_TRK,
            model_complexity=MP_MODEL_COMPLEXITY,
            smooth_landmarks=True
        )
        self.joint_smoother = JointSmoother(SMOOTH_IDX)
        self.pose_buf = PoseBuffer(POSE_BUFFER_N)
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                rgb_roi, bbox, frame_shape = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            sx, sy, sw, sh = bbox
            res = self.pose.process(rgb_roi)
            if res and res.pose_landmarks:
                lm_full = translate_landmarks_to_full(res.pose_landmarks, (sx, sy, sw, sh), frame_shape)
                lm_full = self.joint_smoother.smooth(lm_full, frame_shape)
                label, conf, _ = classify_pose_tpose_armsup(lm_full, frame_shape)
                self.pose_buf.push(label)
                maj_label, maj_cnt, _ = self.pose_buf.majority()
                final_label = maj_label if (maj_label!="Unknown" and maj_cnt>=POSE_STABLE_MIN) else label
                with self.out_lock:
                    self.last_label = final_label
                    self.last_conf  = conf
                    self.last_time  = time.perf_counter()
                    self.last_ok    = True
            else:
                with self.out_lock:
                    self.last_ok = False

# ========= 5) Ana Pipeline Sınıfı (Frame in → (class_id, bbox, conf) out) =========
class MediapipePoseModule:
    """
    Kullanım:
        pipe = MediapipePoseModule()
        cls_id, bbox, conf = pipe.process(frame_bgr)
        ...
        pipe.close()
    """
    def __init__(self):
        # YOLO
        self.yolo_model = YOLO(MODEL_PATH)
        try: self.yolo_model.to(DEVICE)
        except Exception: pass

        # Warm-up (opsiyonel ama stabil)
        try:
            if _HAS_TORCH:
                torch.backends.cudnn.benchmark = True
                try: torch.set_float32_matmul_precision('high')
                except Exception: pass
                dummy = np.zeros((YOLO_IMGSZ_BASE, YOLO_IMGSZ_BASE, 3), dtype=np.uint8)
                with torch.inference_mode():
                    self.yolo_model(dummy, classes=[PERSON_CLASS_ID], device=DEVICE,
                                    imgsz=YOLO_IMGSZ_BASE, half=HALF, verbose=False)
        except Exception:
            pass

        self.roi_stab = RoiStabilizer()
        self.worker = PoseWorker(); self.worker.start()

        # Durum değişkenleri
        self.det_interval = 3
        self.last_yolo_frame = -9999
        self.yolo_imgsz_runtime = YOLO_IMGSZ_BASE
        self.current_pad = BBOX_PADDING_BASE
        self.label_hist = deque(maxlen=POSE_BUFFER_N)
        self.unknown_streak = 0
        self.frame_count = 0

        # Hazır objeler
        self.cvtColor = cv2.cvtColor
        self.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
        self.COLOR_GRAY2RGB = cv2.COLOR_GRAY2RGB
        self.COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    @staticmethod
    def _label_to_class(label: str) -> int:
        if label == "T-POSE": return 0
        if label == "ARMS-UP": return 1
        return -1

    def process(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[Tuple[int,int,int,int]], float]:
        """
        Girdi: BGR frame (numpy)
        Çıktı: (class_id, bbox(x,y,w,h) veya None, conf[0..1])
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return -1, None, 0.0

        self.frame_count += 1
        H, W = frame_bgr.shape[:2]

        # ---- YOLO: adaptif imgsz + adaptif aralık ----
        if (self.frame_count - self.last_yolo_frame) >= self.det_interval or self.roi_stab.state is None:
            bbox, conf_det = yolo_detect_person(self.yolo_model, frame_bgr, self.yolo_imgsz_runtime)
            self.last_yolo_frame = self.frame_count
            if bbox is not None:
                maj_label = self.label_hist[0] if self.label_hist else "Unknown"
                pad_to_use = BBOX_PADDING_TPOSE if maj_label == "T-POSE" else self.current_pad
                x, y, w, h = pad_bbox(*bbox, pad_to_use, W, H)
                self.roi_stab.update((x, y, w, h), frame_bgr.shape)
                short_side = min(w, h)
                self.yolo_imgsz_runtime = YOLO_IMGSZ_HIGH if short_side < SHORT_SIDE_HIGH_IMGSZ_THRESH else YOLO_IMGSZ_BASE

        # ---- ROI varsa worker'a gönder ----
        st = self.roi_stab.state
        if st is not None:
            sx, sy, sw, sh = st
            roi = frame_bgr[sy:sy+sh, sx:sx+sw]
            if roi.size > 0:
                if min(sw, sh) >= MIN_SHORT_SIDE_MP:
                    gray = self.cvtColor(roi, self.COLOR_BGR2GRAY)
                    mean_luma = float(gray.mean())
                    if mean_luma < CLAHE_LUMA_THRESH:
                        eq = self._clahe.apply(gray)
                        rgb_roi = self.cvtColor(eq, self.COLOR_GRAY2RGB)
                    else:
                        rgb_roi = self.cvtColor(roi, self.COLOR_BGR2RGB)
                    try:
                        while not self.worker.in_q.empty():
                            self.worker.in_q.get_nowait()
                        self.worker.in_q.put_nowait((rgb_roi, (sx, sy, sw, sh), frame_bgr.shape))
                    except queue.Full:
                        pass  # bir sonraki karede günceller
        # ---- Worker sonucu oku ----
        with self.worker.out_lock:
            ok_pose  = self.worker.last_ok
            label    = self.worker.last_label
            conf_val = self.worker.last_conf

        if ok_pose:
            self.label_hist.append(label); self.unknown_streak = 0
        else:
            self.label_hist.append("Unknown"); self.unknown_streak += 1

        if len(self.label_hist) >= POSE_BUFFER_N:
            vals, cnts = np.unique(self.label_hist, return_counts=True)
            top_i = int(np.argmax(cnts))
            top_label, top_cnt = vals[top_i], int(cnts[top_i])
            top_ratio = float(top_cnt)/float(len(self.label_hist))
            if top_label != "Unknown" and top_ratio >= LABEL_STABLE_RATIO and self.det_interval < DET_MAX:
                self.det_interval += 1
                self.label_hist.clear()
        if self.unknown_streak >= UNKNOWN_STREAK_RESET:
            self.det_interval = DET_MIN; self.unknown_streak = 0

        # Döndürülecek bbox (varsa ROI state)
        bbox_out: Optional[Tuple[int,int,int,int]] = tuple(self.roi_stab.state) if self.roi_stab.state is not None else None
        class_id = self._label_to_class(label)
        return class_id, bbox_out, float(conf_val)

    def close(self):
        try:
            self.worker.stop()
            self.worker.join(timeout=0.5)
        except Exception:
            pass
