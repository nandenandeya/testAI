"""
train_model_fixed.py - Training model dengan regularisasi dan pembatasan
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

def load_data(csv_path='features_cleaned.csv'):
    """Load dataset yang sudah dibersihkan"""
    df = pd.read_csv(csv_path)
    X = df[["ear", "head_pose", "mouth_ratio"]]
    y = df["label"]
    return X, y, df

def train_improved(X, y):
    """
    Training dengan multiple strategies untuk mengatasi jomplang
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    
    # Strategy 1: RobustScaler + Strong Regularization
    print("\n🤖 Strategy 1: RobustScaler + C=0.5")
    pipeline1 = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,  # Stronger regularization
            random_state=42
        ))
    ])
    pipeline1.fit(X_train, y_train)
    y_pred1 = pipeline1.predict(X_test)
    results['strategy1'] = {
        'pipeline': pipeline1,
        'f1': f1_score(y_test, y_pred1),
        'acc': accuracy_score(y_test, y_pred1),
        'coef': pipeline1.named_steps['clf'].coef_[0]
    }
    
    # Strategy 2: ElasticNet (L1+L2) dengan weight decay
    print("🤖 Strategy 2: ElasticNet + Saga solver")
    pipeline2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            C=0.3,
            penalty='elasticnet',
            l1_ratio=0.5,
            solver='saga',
            random_state=42
        ))
    ])
    pipeline2.fit(X_train, y_train)
    y_pred2 = pipeline2.predict(X_test)
    results['strategy2'] = {
        'pipeline': pipeline2,
        'f1': f1_score(y_test, y_pred2),
        'acc': accuracy_score(y_test, y_pred2),
        'coef': pipeline2.named_steps['clf'].coef_[0]
    }
    
    # Strategy 3: Dengan pembatasan fitur (feature engineering)
    print("🤖 Strategy 3: With feature capping")
    X_capped = X.copy()
    X_capped['mouth_ratio'] = X_capped['mouth_ratio'].clip(upper=0.12)
    X_capped['ear'] = X_capped['ear'].clip(lower=0.10, upper=0.40)
    
    X_train_cap, X_test_cap, y_train_cap, y_test_cap = train_test_split(
        X_capped, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline3 = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.8,
            random_state=42
        ))
    ])
    pipeline3.fit(X_train_cap, y_train_cap)
    y_pred3 = pipeline3.predict(X_test_cap)
    results['strategy3'] = {
        'pipeline': pipeline3,
        'f1': f1_score(y_test_cap, y_pred3),
        'acc': accuracy_score(y_test_cap, y_pred3),
        'coef': pipeline3.named_steps['clf'].coef_[0]
    }
    
    # Pilih model terbaik
    best_strategy = max(results.keys(), key=lambda k: results[k]['f1'])
    best_pipeline = results[best_strategy]['pipeline']
    
    print(f"\n✅ Best model: {best_strategy}")
    print(f"   F1 Score: {results[best_strategy]['f1']:.3f}")
    print(f"   Accuracy: {results[best_strategy]['acc']:.3f}")
    
    return best_pipeline, results[best_strategy]

def save_model(pipeline, filename='focus_model_fixed.pkl'):
    """Simpan model yang sudah dilatih"""
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\n💾 Model saved: {filename}")

def generate_report(pipeline, X_test, y_test, coefs):
    """Generate training report"""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    report = []
    report.append("=" * 60)
    report.append("FOCUSWEBCAM - MODEL TRAINING REPORT (FIXED VERSION)")
    report.append("=" * 60)
    report.append(f"\nAccuracy  : {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.2f}%)")
    report.append(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    report.append(f"\nConfusion Matrix:")
    report.append(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    report.append(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    report.append(f"\nClassification Report:")
    report.append(classification_report(y_test, y_pred, target_names=["TIDAK_FOKUS", "FOKUS"]))
    
    report.append(f"\nModel Coefficients (setelah perbaikan):")
    feature_names = ["ear", "head_pose", "mouth_ratio"]
    for name, coef in zip(feature_names, coefs):
        impact = "↑ meningkatkan FOKUS" if coef > 0 else "↓ menurunkan FOKUS"
        report.append(f"  {name:<15}: {coef:+.4f}  {impact}")
    
    # Cek apakah masih jomplang
    mouth_coef = coefs[2]
    ear_coef = coefs[0]
    ratio = abs(mouth_coef / ear_coef) if ear_coef != 0 else 999
    
    report.append(f"\n📊 Rasio koefisien Mouth/EAR: {ratio:.2f}")
    if ratio > 1.5:
        report.append(f"⚠️  PERINGATAN: Mouth coefficient masih {ratio:.1f}x lebih besar dari EAR!")
        report.append(f"    Pertimbangkan untuk menggunakan ensemble method.")
    else:
        report.append(f"✅ Koefisien sudah seimbang (rasio < 1.5)")
    
    report.append("\n" + "=" * 60)
    
    # Save report
    with open('training_report_fixed.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print("\n".join(report))
    print(f"\n📄 Report saved: training_report_fixed.txt")

def main():
    print("=" * 60)
    print("TRAINING MODEL FOCUSWEBCAM - FIXED VERSION")
    print("=" * 60)
    
    # Load cleaned data
    X, y, df = load_data('features_cleaned.csv')
    print(f"\n📊 Loaded {len(X)} samples")
    print(f"   Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test : {len(X_test)} samples")
    
    # Train best model
    best_pipeline, best_result = train_improved(X_train, y_train)
    
    # Save model
    save_model(best_pipeline, 'focus_model_fixed.pkl')
    
    # Generate report
    generate_report(best_pipeline, X_test, y_test, best_result['coef'])
    
    print("\n🎉 Training selesai! Model siap digunakan.")

if __name__ == "__main__":
    main()