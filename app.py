"""
FocusWebCam — Streamlit App (FIXED VERSION)
Mengatasi masalah skor jomplang dengan post-processing
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
from datetime import datetime

import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page config
st.set_page_config(
    page_title="FocusWebCam | Ethical AI Focus Detection (Fixed)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
NOSE_TIP, FACE_LEFT, FACE_RIGHT = 1, 234, 454

# Model parameters (dari training_report_fixed.txt)
MODEL_COEF = {"ear": 1.0494, "head_pose": -2.6625, "mouth_ratio": 2.0005}
MODEL_INTERCEPT = -0.5234
MODEL_SCALER = {
    "ear":         {"mean": 0.214, "std": 0.098},
    "head_pose":   {"mean": 0.178, "std": 0.245},
    "mouth_ratio": {"mean": 0.068, "std": 0.082},
}

# 🔧 PERBAIKAN: Parameter untuk normalisasi
ALERT_THRESHOLD  = 40
EAR_OPEN         = 0.25
EAR_CLOSED       = 0.15
SMOOTHING_WINDOW = 5  # Increased from 3

# 🔥 PARAMETER BARU UNTUK MENGATASI JOMPLANG
MOUTH_MAX_REALISTIC = 0.12  # Nilai maksimal mouth ratio yang realistis
EAR_MIN_REALISTIC = 0.10
EAR_MAX_REALISTIC = 0.40
HEAD_MAX_REALISTIC = 0.30

# Bobot untuk ensemble scoring (lebih seimbang)
WEIGHT_EAR = 0.50      # Mata: paling penting (dinaikkan)
WEIGHT_HEAD = 0.35     # Posisi kepala
WEIGHT_MOUTH = 0.15    # Mulut: diturunkan drastis

# Fitur helpers
def calc_ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = np.hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
    B = np.hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
    C = np.hypot(pts[0][0]-pts[3][0], pts[0][1]-pts[3][1])
    return (A + B) / (2.0 * C) if C else 0.0

def calc_head_pose(lm, w, h):
    nose  = lm[NOSE_TIP]
    left  = lm[FACE_LEFT]
    right = lm[FACE_RIGHT]
    face_center = (left.x + right.x) / 2
    face_width  = abs(right.x - left.x)
    return abs(nose.x - face_center) / face_width if face_width else 0.0

def calc_mouth(lm, w, h):
    top    = lm[MOUTH_TOP]
    bottom = lm[MOUTH_BOTTOM]
    left   = lm[MOUTH_LEFT]
    right  = lm[MOUTH_RIGHT]
    vertical   = abs((top.y - bottom.y) * h)
    horizontal = abs((left.x - right.x) * w)
    ratio = vertical / horizontal if horizontal else 0.0
    # 🔥 PERBAIKAN: Batasi mouth ratio ke nilai realistis
    return min(ratio, MOUTH_MAX_REALISTIC)

def standardize(v, mean, std):
    return (v - mean) / std if std else 0.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_probability(ear, head_pose, mouth):
    # 🔥 PERBAIKAN: Clamp semua fitur ke range realistis
    ear = max(EAR_MIN_REALISTIC, min(EAR_MAX_REALISTIC, ear))
    head_pose = min(head_pose, HEAD_MAX_REALISTIC)
    mouth = min(mouth, MOUTH_MAX_REALISTIC)
    
    ear_s   = standardize(ear,       **MODEL_SCALER["ear"])
    head_s  = standardize(head_pose, **MODEL_SCALER["head_pose"])
    mouth_s = standardize(mouth,     **MODEL_SCALER["mouth_ratio"])
    
    # Clamp standardized values to prevent extreme values
    ear_s = max(-3, min(3, ear_s))
    head_s = max(-3, min(3, head_s))
    mouth_s = max(-3, min(3, mouth_s))
    
    logit = (MODEL_COEF["ear"]        * ear_s  +
             MODEL_COEF["head_pose"]  * head_s +
             MODEL_COEF["mouth_ratio"]* mouth_s +
             MODEL_INTERCEPT)
    return float(sigmoid(logit))

def calculate_ensemble_score(ear, head_pose, mouth):
    """
    🔥 FUNGSI BARU: Ensemble scoring untuk mengatasi jomplang
    Menggabungkan model LR dengan rule-based scoring
    """
    # 1. Logistic Regression score
    lr_prob = predict_probability(ear, head_pose, mouth)
    lr_score = lr_prob * 100
    
    # 2. Rule-based score (lebih seimbang)
    # Normalisasi EAR (0.10→0, 0.40→100)
    ear_norm = (ear - EAR_MIN_REALISTIC) / (EAR_MAX_REALISTIC - EAR_MIN_REALISTIC)
    ear_norm = max(0, min(1, ear_norm))
    ear_score = ear_norm * 100
    
    # Normalisasi Head Pose (0→100, 0.30→0)
    head_norm = 1 - min(1, head_pose / HEAD_MAX_REALISTIC)
    head_score = head_norm * 100
    
    # Normalisasi Mouth (0→100, 0.12→0)
    mouth_norm = 1 - min(1, mouth / MOUTH_MAX_REALISTIC)
    mouth_score = mouth_norm * 100
    
    # Weighted ensemble
    weighted_score = (
        WEIGHT_EAR * ear_score +
        WEIGHT_HEAD * head_score +
        WEIGHT_MOUTH * mouth_score
    )
    
    # 3. Detect anomalies (jika LR score sangat berbeda)
    if lr_score > 80 and ear < 0.20:
        # Anomali: mulut terbuka menyebabkan LR score tinggi
        # Gunakan weighted score lebih dominan
        final_score = 0.4 * lr_score + 0.6 * weighted_score
    elif lr_score < 40 and ear > 0.25 and mouth < 0.06:
        # Anomali: skor terlalu rendah padahal kondisi normal
        final_score = 0.3 * lr_score + 0.7 * weighted_score
    else:
        # Normal: blend 50-50
        final_score = 0.5 * lr_score + 0.5 * weighted_score
    
    return max(0, min(100, final_score))

def get_color(score):
    if score >= 65:  return (0, 255, 136)
    if score >= 40:  return (0, 200, 255)
    return (80, 80, 255)

def explain_score(ear, head, mouth, score):
    """
    Improved explanation dengan deteksi anomali
    """
    neg = []
    if ear < 0.18: neg.append("mata tertutup/berkedip")
    elif ear > 0.32: neg.append("mata terlalu terbuka (bisa jadi terkejut)")
    
    if head > 0.15: neg.append("kepala menoleh")
    if mouth > 0.08: neg.append("mulut terbuka")
    
    # Deteksi anomali mulut terbuka tapi score tinggi
    if score > 70 and mouth > 0.10 and ear < 0.22:
        return f"⚠️ Skor {score}/100 (perlu koreksi: deteksi mulut terlalu dominan)"
    
    if score >= 65:
        return f"✅ Fokus baik ({score}/100)"
    elif score >= 40:
        isu = ", ".join(neg) if neg else "pertahankan kondisi"
        return f"⚡ Perhatian ({score}/100) — {isu}"
    else:
        isu = ", ".join(neg) if neg else "kondisi tidak optimal"
        return f"⚠️ Tidak fokus ({score}/100) — {isu}"

# Queue untuk kirim data
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=5)

result_queue: queue.Queue = st.session_state.result_queue

# Session state
def init_state():
    defaults = {
        "session_active":   False,
        "session_start":    None,
        "score_history":    [],
        "alert_count":      0,
        "low_score_count":  0,
        "last_alert_time":  0,
        "log_entries":      ["— Sistem siap (Fixed Version) —"],
        "consent_given":    False,
        "consent_asked":    False,
        "disp_score":       None,
        "disp_ear":         None,
        "disp_head":        None,
        "disp_mouth":       None,
        "disp_expl":        "",
        "disp_face":        False,
        "anomaly_corrected": 0,  # Counter untuk anomali yang dikoreksi
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Video Processor dengan ensemble scoring
class FocusVideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smooth = deque(maxlen=SMOOTHING_WINDOW)
        self.correction_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm    = res.multi_face_landmarks[0].landmark
            ear_l = calc_ear(lm, LEFT_EYE,  w, h)
            ear_r = calc_ear(lm, RIGHT_EYE, w, h)
            ear   = (ear_l + ear_r) / 2.0
            head  = calc_head_pose(lm, w, h)
            mouth = calc_mouth(lm, w, h)  # Already capped

            # 🔥 MENGGUNAKAN ENSEMBLE SCORE
            score_raw = calculate_ensemble_score(ear, head, mouth)
            self._smooth.append(score_raw)
            score = int(np.clip(round(np.mean(self._smooth)), 0, 100))
            color = get_color(score)
            expl  = explain_score(ear, head, mouth, score)
            
            # Deteksi koreksi anomali untuk logging
            if (score > 70 and mouth > 0.10 and ear < 0.22) or \
               (score < 40 and ear > 0.25 and mouth < 0.06):
                self.correction_count += 1
                st.session_state.anomaly_corrected = self.correction_count

            # Kirim data
            data = {
                "face":  True,
                "score": score,
                "ear":   round(ear,   4),
                "head":  round(head,  4),
                "mouth": round(mouth, 4),
                "expl":  expl,
            }
            try:
                result_queue.put_nowait(data)
            except queue.Full:
                try:    result_queue.get_nowait()
                except: pass
                try:    result_queue.put_nowait(data)
                except: pass

            # Draw overlay
            for idx in LEFT_EYE + RIGHT_EYE:
                pt = lm[idx]
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 2, color, -1)

            fl = lm[FACE_LEFT]; fr = lm[FACE_RIGHT]
            ft = lm[10];        fb = lm[152]
            cv2.rectangle(img,
                (int(fl.x*w), int(ft.y*h)),
                (int(fr.x*w), int(fb.y*h)),
                color, 1)

            # HUD with correction info
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (230, 115), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
            cv2.putText(img, f"FOCUS: {score}", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(img, f"EAR:{ear:.3f}  HEAD:{head:.3f}", (18, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
            cv2.putText(img, f"MOUTH:{mouth:.3f}", (18, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
            
            # 🔥 Tambahan info metode scoring
            cv2.putText(img, "ENSEMBLE MODE", (18, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,100,100), 1)
            
            bar_w = int((score/100)*182)
            cv2.rectangle(img, (18, 108), (200, 117), (40,40,40), -1)
            cv2.rectangle(img, (18, 108), (18+bar_w, 117), color, -1)

        else:
            cv2.putText(img, "Tidak ada wajah terdeteksi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,80,80), 1)
            data = {"face": False, "score": 0,
                    "ear": None, "head": None, "mouth": None, "expl": ""}
            try:
                result_queue.put_nowait(data)
            except queue.Full:
                try:    result_queue.get_nowait()
                except: pass
                try:    result_queue.put_nowait(data)
                except: pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Drain queue
def drain_queue():
    latest = None
    while True:
        try:
            latest = result_queue.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return False

    st.session_state.disp_score = latest["score"]
    st.session_state.disp_ear   = latest["ear"]
    st.session_state.disp_head  = latest["head"]
    st.session_state.disp_mouth = latest["mouth"]
    st.session_state.disp_expl  = latest["expl"]
    st.session_state.disp_face  = latest["face"]

    if st.session_state.session_active:
        score = latest["score"]
        st.session_state.score_history.append(score)

        if score < ALERT_THRESHOLD:
            st.session_state.low_score_count += 1
        else:
            st.session_state.low_score_count = 0

        now = time.time()
        if (st.session_state.low_score_count >= 5 and
                now - st.session_state.last_alert_time >= 30):
            st.session_state.alert_count += 1
            st.session_state.last_alert_time = now
            st.session_state.low_score_count = 0
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_entries.insert(
                0, f"⚠️ [{ts}] Alert #{st.session_state.alert_count} — skor {score}")

    return True

# CSS (sama seperti sebelumnya, bisa ditambahkan info koreksi)
st.markdown("""
<style>
/* CSS sama seperti sebelumnya, tambahan: */
.score-card-fixed {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 8px;
    position: relative;
}
.badge-ensemble {
    position: absolute;
    top: 8px;
    right: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.45rem;
    color: #00ff88;
    background: rgba(0,255,136,0.1);
    padding: 2px 6px;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

# Consent dialog (sama)
if not st.session_state.consent_asked:
    @st.dialog("📋 Persetujuan Privasi")
    def _consent():
        st.markdown("""
        **FocusWebCam (Fixed Version)** memproses data wajah Anda untuk mendeteksi tingkat fokus.
        - ✅ Semua data diproses **lokal di perangkat Anda**
        - ✅ Video tidak pernah dikirim ke server manapun
        - ✅ Menggunakan **Ensemble Scoring** untuk akurasi lebih baik
        - ✅ Hanya skor agregat yang disimpan di session
        """)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Izinkan", use_container_width=True):
                st.session_state.consent_given = True
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, "✅ Persetujuan diberikan (Fixed Version)")
                st.rerun()
        with c2:
            if st.button("❌ Tolak", use_container_width=True):
                st.session_state.consent_given = False
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, "❌ Persetujuan ditolak")
                st.rerun()
    _consent()

# Header
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown('<div class="app-title"><span class="logo-dot"></span>FocusWebCam <span style="font-size:0.6rem;color:#00ff88;">FIXED v2</span></div>',
                unsafe_allow_html=True)
with hc2:
    if st.session_state.session_active:
        status = "SESI AKTIF (ENSEMBLE)"
    elif st.session_state.consent_given:
        status = "SIAP — Ensemble Mode"
    else:
        status = "MODE TERBATAS"
    st.markdown(f'<div class="hstatus">{status}</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

drain_queue()

# Layout
cam_col, info_col = st.columns([3, 2])

with cam_col:
    if not st.session_state.session_active:
        if st.button("▶  MULAI SESI (ENSEMBLE)", key="btn_start"):
            st.session_state.session_active  = True
            st.session_state.session_start   = time.time()
            st.session_state.score_history   = []
            st.session_state.alert_count     = 0
            st.session_state.low_score_count = 0
            st.session_state.last_alert_time = 0
            st.session_state.disp_score      = None
            st.session_state.disp_ear        = None
            st.session_state.disp_head       = None
            st.session_state.disp_mouth      = None
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_entries.insert(0, f"🎯 [{ts}] Sesi dimulai (Ensemble Scoring)")
            st.rerun()
    else:
        if st.button("⏹  HENTIKAN SESI", key="btn_stop"):
            hist = st.session_state.score_history
            if hist:
                avg = round(sum(hist)/len(hist))
                pct = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100)
                ts  = datetime.now().strftime("%H:%M:%S")
                st.session_state.log_entries.insert(
                    0, f"📊 [{ts}] Selesai — avg {avg}, fokus {pct}%, {st.session_state.alert_count} alert, {st.session_state.anomaly_corrected} koreksi")
            st.session_state.session_active = False
            st.rerun()

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    ctx = webrtc_streamer(
        key="focus-cam-fixed",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=FocusVideoProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    if not st.session_state.session_active:
        st.info("Tekan **MULAI SESI** setelah kamera aktif. Menggunakan Ensemble Scoring untuk akurasi lebih baik.", icon="🎯")

with info_col:
    score = st.session_state.disp_score
    ear   = st.session_state.disp_ear
    head  = st.session_state.disp_head
    mouth = st.session_state.disp_mouth
    expl  = st.session_state.disp_expl

    if score is not None:
        color_hex = "#00ff88" if score >= 65 else ("#ffcc00" if score >= 40 else "#ff4444")
        state_txt = "FOKUS" if score >= 65 else ("PERHATIAN" if score >= 40 else "TIDAK FOKUS")
        score_disp = str(score)
    else:
        color_hex, state_txt, score_disp = "#555555", "—", "--"
        score = 0

    st.markdown(f"""
    <div class="score-card" style="position:relative;">
      <div class="badge-ensemble">ENSEMBLE</div>
      <div class="score-label">FOCUS SCORE</div>
      <div class="score-big" style="color:{color_hex}">{score_disp}
        <span style="font-size:.9rem;color:#555"> /100</span>
      </div>
      <div class="score-state" style="color:{color_hex}">{state_txt}</div>
    </div>""", unsafe_allow_html=True)

    st.progress(int(score) / 100 if score else 0)

    # Feature cards
    f1, f2, f3 = st.columns(3)
    ed = f"{ear:.3f}"   if ear   is not None else "—"
    hd = f"{head:.3f}"  if head  is not None else "—"
    md = f"{mouth:.3f}" if mouth is not None else "—"

    with f1:
        st.markdown(f'<div class="feat-card"><div class="feat-icon">👁</div>'
                    f'<div class="feat-name">EAR (MATA)</div>'
                    f'<div class="feat-val">{ed}</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown(f'<div class="feat-card"><div class="feat-icon">↔</div>'
                    f'<div class="feat-name">HEAD POSE</div>'
                    f'<div class="feat-val">{hd}</div></div>', unsafe_allow_html=True)
    with f3:
        st.markdown(f'<div class="feat-card"><div class="feat-icon">💬</div>'
                    f'<div class="feat-name">MOUTH RATIO</div>'
                    f'<div class="feat-val">{md}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if expl:
        st.markdown(f'<div class="expl-card">📊 {expl}</div>', unsafe_allow_html=True)
    
    # Info koreksi anomali
    if st.session_state.anomaly_corrected > 0:
        st.caption(f"🔧 {st.session_state.anomaly_corrected} anomali skor telah dikoreksi")

    # Session stats
    hist = st.session_state.score_history
    avg_s = round(sum(hist)/len(hist)) if hist else 0
    fpct  = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100) if hist else 0
    elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
    mm = str(elapsed // 60).zfill(2)
    ss = str(elapsed  % 60).zfill(2)

    st.markdown(f"""
    <div class="stats-card">
      <div class="stats-title">SESI INI</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;text-align:center">
        <div><div class="stat-val">{mm}:{ss}</div><div class="stat-lbl">Durasi</div></div>
        <div><div class="stat-val">{avg_s if hist else "--"}</div><div class="stat-lbl">Rata-rata</div></div>
        <div><div class="stat-val">{fpct if hist else "--"}%</div><div class="stat-lbl">Fokus</div></div>
        <div><div class="stat-val">{st.session_state.alert_count}</div><div class="stat-lbl">Alert</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Log
    st.markdown('<div class="log-card"><div class="log-title">LOG AKTIVITAS</div>', unsafe_allow_html=True)
    logs_html = ""
    for entry in st.session_state.log_entries[:20]:
        cls = ("log-alert" if any(x in entry for x in ["⚠️","❌"])
               else "log-focus" if any(x in entry for x in ["✅","🎯","📊"])
               else "log-sys")
        logs_html += f'<div class="log-entry {cls}">{entry}</div>'
    st.markdown(logs_html + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="privacy-note">🔒 Data diproses lokal | 🎯 Ensemble Scoring (LR + Rule-based)</div>',
                unsafe_allow_html=True)

# Auto-refresh
if ctx.state.playing:
    time.sleep(0.5)
    st.rerun()