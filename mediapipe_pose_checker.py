# mediaipipe_pose_checker.py

# BLAS/OpenMP oversubscription engellemek için konuldu.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")

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

# Model olarak M modeli seçildi.
MODEL_PATH = 'models/yolov8m.pt'
PERSON_CLASS_ID = 0

# Eşikler ve parametreler
MIN_YOLO_CONF = 0.55 # YOLO'nun insan kabul ettiği min güven skoru.
# Dinamik padding
BBOX_PADDING_BASE  = 45 # MP'ye verilecek box'a padding veriyoruz.
BBOX_PADDING_TPOSE = 70 # T-Pose daha geniş yer kaplayacağı için daha fazla padding veriyoruz.
# MediaPipe
MP_MIN_DET = 0.60 # MP'nin ilk tespitindeki güven eşiği.
MP_MIN_TRK = 0.50 # MP'nin ilkten sonraki tespitlerinde güven eşiği.
MP_MODEL_COMPLEXITY = 1 # Burayla oynamak sağlıklı değil.
# CLAHE tetikleyici
CLAHE_LUMA_THRESH = 50.0  # Clahe, kontrastı arttırarak MP'ye vermemizi sağlar.

# Torch var mı?
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

DEVICE = "cuda:0" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu" # Destekleyici CUDA
HALF = DEVICE.startswith("cuda")

# Dinamik YOLO imgsz
YOLO_IMGSZ_BASE = 448
YOLO_IMGSZ_HIGH = 640
SHORT_SIDE_HIGH_IMGSZ_THRESH = 100
MIN_SHORT_SIDE_MP = 64 # Eğer 64 px'den düşükse göremeyeceği için MP tarafını uğraştırmıyoruz.

# ROI stabilizasyon parametreleri.
ROI_SMOOTH_ALPHA = 0.5
ROI_MAX_CENTER_JUMP_PCT = 0.035
ROI_MAX_SIZE_JUMP_PCT   = 0.04

# One-Euro parametreleri. (Daha yumuşak eklem akışı ile daha iyi tespit sağlamak için.)
SMOOTH_IDX = [13, 14, 15, 16] # Bu işlemin uygulanacağı eklem noktaları.
ONE_EURO_FREQUENCY   = 30.0
ONE_EURO_MINCUTOFF   = 0.8
ONE_EURO_BETA        = 0.015
ONE_EURO_DCUT        = 1.0

DET_MIN, DET_MAX = 1, 8
LABEL_STABLE_RATIO = 0.70
UNKNOWN_STREAK_RESET = 3

# Pose eşikleri (geometri tarafı için.)
POSE_MIN_VIS           = 0.5 # MP'nin bir eklemi hesaba dahil edebilmesi için gerekli min güven eşiği.
POSE_BUFFER_N          = 9 # Karar vermeden önce, en son 9 frame'i buffer'a al.
POSE_STABLE_MIN        = 3 # O pozu kararlı sayabilmek için en az 3 karede görülmeli.
TPOSE_SPINE_MIN, TPOSE_SPINE_MAX = 70.0, 115.0 # Omurga ile yapılan açı TPOSE için 70 ve 105 arası seçildi.
TPOSE_ELBOW_MIN = 155.0 # Ayrıca dirseğin sadece kendisini de hesaba katıyoruz.
ARMSUP_SPINE_MAX = 45.0 # Omurga ile yapılan açı ARMSUP için 45 ve 125 arası olarak seçildi.
ARMSUP_ELBOW_MIN = 125.0

TOPK_PERSONS       = 2 # Görevde 2 manken olduğu için YOLO'dan en güvenilir maksimum 2 kareyi alabileceğini söylüyoruz.
ASSIGN_IOU_MIN     = 0.15
ASSIGN_CENTER_MAXF = 0.12
TRACK_AGE_MAX      = 25 # Görünmeyen tracker kapanır.

DEDUP_IOU_MIN          = 0.60
DEDUP_CENTER_FRAC      = 0.06
DEDUP_CONTAIN_FRAC     = 0.85
DEDUP_SIZE_SIM_RATIO   = 0.75

SPAWN_DUP_IOU          = 0.55
SPAWN_DUP_CENTER_FRAC  = 0.06 # Merkez çok yakınsa spawn etmiyoruz.
SPAWN_CONFIRM_FRAMES   = 2
SPAWN_PENDING_TTL      = 3

TRACK_SUPPRESS_IOU     = 0.65
TRACK_SUPPRESS_CENTERF = 0.05

# OpenCV ayarları
cv2.setUseOptimized(True)
try: cv2.setNumThreads(0)
except Exception: pass
try: cv2.ocl.setUseOpenCL(False)
except Exception: pass

def clip_bbox(x, y, w, h, W, H) -> Tuple[int, int, int, int]:
    x = max(0, min(int(round(x)), W - 1))
    y = max(0, min(int(round(y)), H - 1))
    w = max(1, min(int(round(w)), W - x))
    h = max(1, min(int(round(h)), H - y))
    return x, y, w, h

def pad_bbox(x, y, w, h, pad, W, H) -> Tuple[int, int, int, int]:
    return clip_bbox(x - pad, y - pad, w + 2*pad, h + 2*pad, W, H)

def _iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw*ah + bw*bh - inter
    return float(inter) / float(union) if union > 0 else 0.0

def _center_dist(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ac = (ax + aw * 0.5, ay + ah * 0.5)
    bc = (bx + bw * 0.5, by + bh * 0.5)
    dx, dy = ac[0]-bc[0], ac[1]-bc[1]
    return math.hypot(dx, dy)

def _containment_ratio(inner, outer) -> float:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    ix2, iy2 = ix+iw, iy+ih
    ox2, oy2 = ox+ow, oy+oh
    x1, y1 = max(ix, ox), max(iy, oy)
    x2, y2 = min(ix2, ox2), min(iy2, oy2)
    w, h = max(0, x2-x1), max(0, y2-y1)
    inter = w*h
    iarea = max(1, iw*ih)
    return float(inter)/float(iarea)

def _size_similarity(a, b) -> float:
    aw, ah = a[2], a[3]; bw, bh = b[2], b[3]
    rw = min(aw/bw, bw/aw) if aw>0 and bw>0 else 0.0
    rh = min(ah/bh, bh/ah) if ah>0 and bh>0 else 0.0
    return 0.5*(rw+rh)

def _dedup_filter(picked: List[Tuple[Tuple[int,int,int,int], float]],
                  cand_bbox: Tuple[int,int,int,int],
                  frame_shape) -> bool:
    H, W = frame_shape[:2]
    cthr = DEDUP_CENTER_FRAC * min(W, H)
    for pb, _ in picked:
        iou = _iou_xywh(pb, cand_bbox)
        if iou >= DEDUP_IOU_MIN:
            return True
        cdist = _center_dist(pb, cand_bbox)
        if cdist <= cthr:
            contain = max(_containment_ratio(cand_bbox, pb), _containment_ratio(pb, cand_bbox))
            if contain >= DEDUP_CONTAIN_FRAC:
                return True
            if _size_similarity(pb, cand_bbox) >= DEDUP_SIZE_SIM_RATIO:
                return True
    return False

def select_topk_person_bboxes(results, frame_shape, k=TOPK_PERSONS, min_conf=MIN_YOLO_CONF):
    H, W = frame_shape[:2]
    cands: List[Tuple[Tuple[int,int,int,int], float, float]] = []  # (bbox, conf, score)
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
            cands.append(((int(x1), int(y1), int(w), int(h)), conf, score))
    if not cands:
        return []

    cands.sort(key=lambda t: t[2], reverse=True)

    picked: List[Tuple[Tuple[int,int,int,int], float]] = []
    for (bb, conf, _score) in cands:
        if _dedup_filter(picked, bb, frame_shape):
            continue
        picked.append((bb, conf))
        if len(picked) >= k:
            break
    return picked

def yolo_detect_persons(model, frame, imgsz: int, device=DEVICE, half=HALF):
    try:
        if _HAS_TORCH:
            with torch.inference_mode():
                results = model(
                    frame, classes=[PERSON_CLASS_ID],
                    device=device, imgsz=imgsz, half=half, verbose=False,
                    iou=0.5  # biraz daha agresif NMS
                )
        else:
            results = model(frame, classes=[PERSON_CLASS_ID], imgsz=imgsz, verbose=False, iou=0.5)
    except TypeError:
        results = model(frame, classes=[PERSON_CLASS_ID], imgsz=imgsz, verbose=False)
    return select_topk_person_bboxes(results, frame.shape, k=TOPK_PERSONS, min_conf=MIN_YOLO_CONF)

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
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = max(1e-6, t - self.t_prev)
        freq = 1.0 / dt
        dx = (x - self.x_prev) * freq

        a_d = self.alpha(self.dcutoff, freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff, freq)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
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
    # Buradaki parametreler oynamaya müsait, conf ağırlık hesaplamasına bakılacak.
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
    if core_conf < 0.40:
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

    t_spread_score   = float(np.clip((spread_x_wrist-0.22)/0.20,0.0,1.0))
    t_y_align_score  = float(np.clip(1.0 - abs(y_wrist_mean - y_shoulder_mean)/0.07,0.0,1.0))
    t2d = 0.7*t_spread_score + 0.3*t_y_align_score

    up_vertical_score     = float(np.clip(((y_head_norm - y_wrist_mean)-0.03)/0.10,0.0,1.0))
    up_spread_small_score = float(np.clip(1.0 - (spread_x_wrist - 1.1*max(1e-6,spread_x_shoulder))/0.15,0.0,1.0))
    up2d = 0.7*up_vertical_score + 0.3*up_spread_small_score

    if (t2d-up2d)>0.10 and t2d>=0.45:
        label_2d="T-POSE"; two_d_score=t2d
    elif (up2d-t2d)>0.10 and up2d>=0.45:
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
        conf *= 0.96
    if label in ("T-POSE","ARMS-UP") and upper_vis_avg>=0.60:
        conf = max(conf, 0.5)
    conf = float(np.clip(conf,0.0,1.0))

    dbg = {"core_conf":core_conf,"upper_vis_avg":upper_vis_avg,"wrists_vis_avg":wrists_vis_avg,
           "geom_pair":float(np.clip(geom_pair,0.0,1.0)),
           "two_d":{"t2d":t2d,"up2d":up2d,"spread_x_wrist":spread_x_wrist,"spread_x_shoulder":spread_x_shoulder,
                    "y_wrist_mean":y_wrist_mean,"y_shoulder_mean":y_shoulder_mean,"y_head_norm":y_head_norm},
           "labels":{"geom":label_geom,"two_d":label_2d,"final":label}}
    return label, conf, dbg

mp_pose = mp.solutions.pose

class _TrackerPoseWorker(threading.Thread):
    def __init__(self, mp_cfg: dict, smooth_idx: List[int]):
        super().__init__(name="TrackerPoseWorker", daemon=True)
        self.in_q: "queue.Queue[tuple]" = queue.Queue(maxsize=1)
        self.out_lock = threading.Lock()
        self.last_label = "Unknown"
        self.last_conf  = 0.0
        self.last_ok    = False
        self._stop = threading.Event()
        self.pose = mp_pose.Pose(**mp_cfg)
        self.joint_smoother = JointSmoother(smooth_idx)
        self.pose_buf = PoseBuffer(POSE_BUFFER_N)

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
                label, conf, dbg = classify_pose_tpose_armsup(lm_full, frame_shape)

                # DEBUG İÇİNDİR!!!
                # if label == "Unknown" and conf < 0.5 and isinstance(dbg, dict):
                #     reason = dbg.get("reason", None)
                #     labels = dbg.get("labels", {}) if isinstance(dbg.get("labels", {}), dict) else {}
                #     two_d  = dbg.get("two_d", {}) if isinstance(dbg.get("two_d", {}), dict) else {}
                #     core_conf = dbg.get("core_conf", None)
                #
                #     print_data = {
                #         "reason": reason if isinstance(reason, str) else None,
                #         "core_visibility": f"{core_conf:.2f}" if isinstance(core_conf, (int, float)) else None,
                #         "geom_label": labels.get("geom"),
                #         "two_d_label": labels.get("two_d"),
                #         "t2d_score": f"{two_d.get('t2d'):.2f}" if isinstance(two_d.get("t2d"), (int, float)) else None,
                #         "up2d_score": f"{two_d.get('up2d'):.2f}" if isinstance(two_d.get("up2d"), (int, float)) else None,
                #     }
                #     try:
                #         print(f"[DEBUG - TRACKER] Pose Unknown (low conf). Details: {json.dumps(print_data, ensure_ascii=False)}")
                #     except Exception:
                #         # json serileştirme sorunlarına karşı fallback
                #         print(f"[DEBUG - TRACKER] Pose Unknown (low conf). Details (raw): {print_data}")

                self.pose_buf.push(label)
                maj_label, maj_cnt, _ = self.pose_buf.majority()
                final_label = maj_label if (maj_label!="Unknown" and maj_cnt>=POSE_STABLE_MIN) else label
                with self.out_lock:
                    self.last_label = final_label
                    self.last_conf  = conf
                    self.last_ok    = True
            else:
                # DEBUG İÇİNDİR!!!
                try:
                    os.makedirs("debug_fails", exist_ok=True)
                    fail_path = os.path.join("debug_fails", f"fail_{time.time_ns()}.png")
                    cv2.imwrite(fail_path, rgb_roi)
                    print(f"[DEBUG - TRACKER] MediaPipe found NO landmarks. Failed ROI saved to: {fail_path}")
                except Exception as e:
                    print(f"[DEBUG - TRACKER] Failed to save ROI: {e}")
                with self.out_lock:
                    self.last_ok = False

class PoseTracker:
    def __init__(self, track_id: int, init_bbox: Tuple[int,int,int,int], frame_shape, mp_cfg: dict):
        self.track_id = track_id
        self.bbox = init_bbox
        self.age = 0
        self.frame_shape = frame_shape
        self.roi_stab = RoiStabilizer()
        self.bbox = self.roi_stab.update(init_bbox, frame_shape)
        self.worker = _TrackerPoseWorker(mp_cfg=mp_cfg, smooth_idx=SMOOTH_IDX)
        self.worker.start()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self._last_label_for_pad = "Unknown"

    @staticmethod
    def _label_to_class(label: str) -> int:
        if label == "T-POSE": return 0
        if label == "ARMS-UP": return 1
        return -1

    def update_detection(self, det_bbox: Tuple[int,int,int,int], frame_shape):
        self.frame_shape = frame_shape
        # T-POSE iken biraz daha geniş pad kullan
        pad = BBOX_PADDING_TPOSE if self._last_label_for_pad == "T-POSE" else BBOX_PADDING_BASE
        W, H = frame_shape[1], frame_shape[0]
        x, y, w, h = pad_bbox(*det_bbox, pad, W, H)
        self.bbox = self.roi_stab.update((x, y, w, h), frame_shape)
        self.age = 0

    def mark_missed(self):
        self.age += 1

    def process_on_frame(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[Tuple[int,int,int,int]], float]:
        if self.bbox is None:
            return -1, None, 0.0
        sx, sy, sw, sh = self.bbox
        H, W = frame_bgr.shape[:2]
        sx, sy, sw, sh = clip_bbox(sx, sy, sw, sh, W, H)
        if sw <= 0 or sh <= 0:
            return -1, None, 0.0
        roi = frame_bgr[sy:sy+sh, sx:sx+sw]
        if roi.size == 0:
            return -1, None, 0.0
        if min(sw, sh) < MIN_SHORT_SIDE_MP:
            return -1, (sx, sy, sw, sh), 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_luma = float(gray.mean())
        if mean_luma < CLAHE_LUMA_THRESH:
            eq = self._clahe.apply(gray)
            rgb_roi = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
        else:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        try:
            while not self.worker.in_q.empty():
                self.worker.in_q.get_nowait()
            self.worker.in_q.put_nowait((rgb_roi, (sx, sy, sw, sh), frame_bgr.shape))
        except queue.Full:
            pass

        with self.worker.out_lock:
            ok_pose  = self.worker.last_ok
            label    = self.worker.last_label
            conf_val = self.worker.last_conf

        if ok_pose:
            self._last_label_for_pad = label
        class_id = self._label_to_class(label)
        return class_id, (sx, sy, sw, sh), float(conf_val)

    def close(self):
        try:
            self.worker.stop()
            self.worker.join(timeout=0.5)
        except Exception:
            pass

class MultiPoseManager:
    def __init__(self, topk: int = TOPK_PERSONS):
        self.yolo_model = YOLO(MODEL_PATH)
        try: self.yolo_model.to(DEVICE)
        except Exception: pass

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

        self.topk = max(1, int(topk))
        self.yolo_imgsz_runtime = YOLO_IMGSZ_BASE

        self.trackers: Dict[int, PoseTracker] = {}
        self.next_id = 0

        self.iou_min = ASSIGN_IOU_MIN
        self.center_max_frac = ASSIGN_CENTER_MAXF

        self.pending_spawn: List[Dict] = []

    def _too_similar_to_any_tracker(self, bbox, frame_shape) -> bool:
        H, W = frame_shape[:2]
        cthr = SPAWN_DUP_CENTER_FRAC * min(W, H)
        for tid, tr in self.trackers.items():
            if tr.bbox is None:
                continue
            if _iou_xywh(tr.bbox, bbox) >= SPAWN_DUP_IOU:
                return True
            if _center_dist(tr.bbox, bbox) <= cthr:
                return True
        return False

    def _update_pending_spawn(self, unmatched_dets: List[Tuple[Tuple[int,int,int,int], float]], frame_shape):
        H, W = frame_shape[:2]
        cthr = SPAWN_DUP_CENTER_FRAC * min(W, H)

        new_pending: List[Dict] = []
        for p in self.pending_spawn:
            p["ttl"] -= 1
            if p["ttl"] > 0:
                new_pending.append(p)
        self.pending_spawn = new_pending

        for dbox, _conf in unmatched_dets:
            matched_idx = -1
            best_dist = 1e9
            for i, p in enumerate(self.pending_spawn):
                dist = _center_dist(p["bbox"], dbox)
                if dist < best_dist and dist <= cthr:
                    best_dist = dist; matched_idx = i
            if matched_idx >= 0:
                self.pending_spawn[matched_idx]["bbox"] = dbox
                self.pending_spawn[matched_idx]["seen"] += 1
                self.pending_spawn[matched_idx]["ttl"]  = SPAWN_PENDING_TTL
            else:
                self.pending_spawn.append({"bbox": dbox, "seen": 1, "ttl": SPAWN_PENDING_TTL})

        spawn_list = []
        kept_pending = []
        for p in self.pending_spawn:
            if p["seen"] >= SPAWN_CONFIRM_FRAMES and not self._too_similar_to_any_tracker(p["bbox"], frame_shape):
                spawn_list.append(p["bbox"])
            else:
                kept_pending.append(p)
        self.pending_spawn = kept_pending
        return spawn_list

    def _spawn_tracker(self, det_bbox, frame_shape):
        tid = self.next_id; self.next_id += 1
        mp_cfg = dict(
            min_detection_confidence=MP_MIN_DET,
            min_tracking_confidence=MP_MIN_TRK,
            model_complexity=MP_MODEL_COMPLEXITY,
            smooth_landmarks=True
        )
        tr = PoseTracker(tid, det_bbox, frame_shape, mp_cfg)
        self.trackers[tid] = tr
        return tr

    def _assign_detections(self, dets: List[Tuple[Tuple[int,int,int,int], float]], frame_shape):
        if not dets and not self.trackers:
            return

        H, W = frame_shape[:2]
        cmax = self.center_max_frac * min(W, H)

        tracker_ids = list(self.trackers.keys())
        tr_boxes = [self.trackers[tid].bbox for tid in tracker_ids]
        used_det = set()
        used_tr  = set()

        # Skor listesi: (score, tidx, didx)
        pairs = []
        for ti, tbox in enumerate(tr_boxes):
            if tbox is None:
                continue
            for di, (dbox, _conf) in enumerate(dets):
                iou = _iou_xywh(tbox, dbox)
                if iou < self.iou_min:
                    continue
                cdist = _center_dist(tbox, dbox)
                if cdist > cmax:
                    continue
                score = iou - 0.001*(cdist / max(cmax,1e-6))
                pairs.append((score, ti, di))
        pairs.sort(key=lambda t: t[0], reverse=True)

        for _score, ti, di in pairs:
            if ti in used_tr or di in used_det:
                continue
            tid = tracker_ids[ti]
            self.trackers[tid].update_detection(dets[di][0], frame_shape)
            used_tr.add(ti); used_det.add(di)

        unmatched = [(dbox, conf) for di, (dbox, conf) in enumerate(dets) if di not in used_det]

        unmatched_filtered = [(bb, cf) for (bb, cf) in unmatched if not self._too_similar_to_any_tracker(bb, frame_shape)]
        to_spawn = self._update_pending_spawn(unmatched_filtered, frame_shape)
        for bb in to_spawn:
            self._spawn_tracker(bb, frame_shape)

        for ti, tid in enumerate(tracker_ids):
            if ti in used_tr:
                continue
            tr = self.trackers.get(tid)
            if tr is not None:
                tr.mark_missed()

        dead = [tid for tid, tr in self.trackers.items() if tr.age > TRACK_AGE_MAX]
        for tid in dead:
            self.trackers[tid].close()
            del self.trackers[tid]

    def _suppress_overlapping_trackers(self, frame_shape):
        tids = list(self.trackers.keys())
        H, W = frame_shape[:2]
        cthr = TRACK_SUPPRESS_CENTERF * min(W, H)
        to_kill = set()
        for i in range(len(tids)):
            for j in range(i+1, len(tids)):
                ti, tj = tids[i], tids[j]
                tri, trj = self.trackers[ti], self.trackers[tj]
                if tri.bbox is None or trj.bbox is None:
                    continue
                iou = _iou_xywh(tri.bbox, trj.bbox)
                if iou >= TRACK_SUPPRESS_IOU and _center_dist(tri.bbox, trj.bbox) <= cthr:
                    kill = ti if tri.age < trj.age else tj
                    to_kill.add(kill)
        for k in to_kill:
            self.trackers[k].close()
            del self.trackers[k]

    def process(self, frame_bgr: np.ndarray) -> List[Tuple[int, Optional[Tuple[int,int,int,int]], float]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        dets = yolo_detect_persons(self.yolo_model, frame_bgr, self.yolo_imgsz_runtime)
        if dets:
            short_min = min(min(bb[0][2], bb[0][3]) for bb in dets)
            self.yolo_imgsz_runtime = YOLO_IMGSZ_HIGH if short_min < SHORT_SIDE_HIGH_IMGSZ_THRESH else YOLO_IMGSZ_BASE

        self._assign_detections(dets, frame_bgr.shape)

        self._suppress_overlapping_trackers(frame_bgr.shape)

        results: List[Tuple[int, Optional[Tuple[int,int,int,int]], float]] = []
        for tid, tr in list(self.trackers.items()):
            cls_id, bbox, conf = tr.process_on_frame(frame_bgr)
            if bbox is not None:
                results.append((cls_id, bbox, conf))

        # İsteğe bağlı soldan sağa sıralama:
        # results.sort(key=lambda r: (r[1][0] if r[1] else 1e9))

        return results

    def process_top1(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[Tuple[int,int,int,int]], float]:
        res = self.process(frame_bgr)
        return res[0] if res else (-1, None, 0.0)

    def close(self):
        for tid in list(self.trackers.keys()):
            try:
                self.trackers[tid].close()
            except Exception:
                pass
        self.trackers.clear()
