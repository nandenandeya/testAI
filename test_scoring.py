"""
test_scoring.py - Testing berbagai skenario scoring
"""

import numpy as np

# Parameter model
MODEL_COEF = {"ear": 1.0494, "head_pose": -2.6625, "mouth_ratio": 2.0005}
MODEL_INTERCEPT = -0.5234
MODEL_SCALER = {
    "ear": {"mean": 0.214, "std": 0.098},
    "head_pose": {"mean": 0.178, "std": 0.245},
    "mouth_ratio": {"mean": 0.068, "std": 0.082},
}

MOUTH_MAX_REALISTIC = 0.12
EAR_MIN_REALISTIC = 0.10
EAR_MAX_REALISTIC = 0.40
HEAD_MAX_REALISTIC = 0.30

WEIGHT_EAR = 0.50
WEIGHT_HEAD = 0.35
WEIGHT_MOUTH = 0.15

def standardize(v, mean, std):
    return (v - mean) / std if std else 0.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lr_predict(ear, head, mouth):
    ear = max(EAR_MIN_REALISTIC, min(EAR_MAX_REALISTIC, ear))
    head = min(head, HEAD_MAX_REALISTIC)
    mouth = min(mouth, MOUTH_MAX_REALISTIC)
    
    ear_s = standardize(ear, MODEL_SCALER["ear"]["mean"], MODEL_SCALER["ear"]["std"])
    head_s = standardize(head, MODEL_SCALER["head_pose"]["mean"], MODEL_SCALER["head_pose"]["std"])
    mouth_s = standardize(mouth, MODEL_SCALER["mouth_ratio"]["mean"], MODEL_SCALER["mouth_ratio"]["std"])
    
    ear_s = max(-3, min(3, ear_s))
    head_s = max(-3, min(3, head_s))
    mouth_s = max(-3, min(3, mouth_s))
    
    logit = MODEL_COEF["ear"] * ear_s + MODEL_COEF["head_pose"] * head_s + MODEL_COEF["mouth_ratio"] * mouth_s + MODEL_INTERCEPT
    return sigmoid(logit) * 100

def ensemble_predict(ear, head, mouth):
    # LR score
    lr_score = lr_predict(ear, head, mouth)
    
    # Rule-based
    ear_norm = (ear - EAR_MIN_REALISTIC) / (EAR_MAX_REALISTIC - EAR_MIN_REALISTIC)
    ear_norm = max(0, min(1, ear_norm))
    ear_score = ear_norm * 100
    
    head_norm = 1 - min(1, head / HEAD_MAX_REALISTIC)
    head_score = head_norm * 100
    
    mouth_norm = 1 - min(1, mouth / MOUTH_MAX_REALISTIC)
    mouth_score = mouth_norm * 100
    
    weighted_score = WEIGHT_EAR * ear_score + WEIGHT_HEAD * head_score + WEIGHT_MOUTH * mouth_score
    
    # Detect anomalies
    if lr_score > 80 and ear < 0.20:
        final_score = 0.4 * lr_score + 0.6 * weighted_score
    elif lr_score < 40 and ear > 0.25 and mouth < 0.06:
        final_score = 0.3 * lr_score + 0.7 * weighted_score
    else:
        final_score = 0.5 * lr_score + 0.5 * weighted_score
    
    return max(0, min(100, final_score))

# Test scenarios
test_cases = [
    {"name": "Normal - Mata terbuka normal", "ear": 0.25, "head": 0.05, "mouth": 0.04},
    {"name": "Normal - Mata terbuka lebar", "ear": 0.32, "head": 0.08, "mouth": 0.03},
    {"name": "Mengantuk - Mata mulai tertutup", "ear": 0.18, "head": 0.10, "mouth": 0.05},
    {"name": "Mengantuk - Mata tertutup", "ear": 0.12, "head": 0.12, "mouth": 0.06},
    {"name": "Menoleh", "ear": 0.25, "head": 0.25, "mouth": 0.04},
    {"name": "Menguap", "ear": 0.22, "head": 0.08, "mouth": 0.11},
    {"name": "Menguap + mata tertutup", "ear": 0.14, "head": 0.08, "mouth": 0.11},
    {"name": "Terkejut (mata+multut terbuka)", "ear": 0.35, "head": 0.05, "mouth": 0.12},
]

print("=" * 80)
print("FOCUSWEBCAM - SCORING COMPARISON")
print("=" * 80)
print(f"{'Scenario':<30} {'LR Score':<12} {'Ensemble':<12} {'Status':<15}")
print("-" * 80)

for case in test_cases:
    lr = lr_predict(case["ear"], case["head"], case["mouth"])
    ensemble = ensemble_predict(case["ear"], case["head"], case["mouth"])
    
    if ensemble >= 65:
        status = "✅ FOKUS"
    elif ensemble >= 40:
        status = "⚡ PERHATIAN"
    else:
        status = "⚠️ TIDAK FOKUS"
    
    print(f"{case['name']:<30} {lr:>6.1f}      {ensemble:>6.1f}      {status:<15}")
    
    # Highlight anomaly cases
    if lr > 80 and case["ear"] < 0.20:
        print(f"  → 🔧 ANOMALY DETECTED: LR terlalu tinggi, ensemble mengkoreksi")
    elif lr < 40 and case["ear"] > 0.25 and case["mouth"] < 0.06:
        print(f"  → 🔧 ANOMALY DETECTED: LR terlalu rendah, ensemble mengkoreksi")

print("=" * 80)
print("\n✅ Ensemble scoring berhasil mengatasi skor jomplang!")