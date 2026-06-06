"""
clean_dataset.py - Membersihkan dataset dari outlier dan nilai tidak realistis
"""

import pandas as pd
import numpy as np
from scipy import stats

def clean_and_save_dataset():
    print("=" * 50)
    print("MEMBERSIHKAN DATASET FOCUSWEBCAM")
    print("=" * 50)
    
    df = pd.read_csv('features.csv')
    original_len = len(df)
    
    print(f"\n📊 Data awal: {original_len} baris")
    print(f"   Label 0 (Tidak Fokus): {(df['label']==0).sum()}")
    print(f"   Label 1 (Fokus): {(df['label']==1).sum()}")
    
    # 1. Filter outlier berdasarkan domain knowledge (realistis)
    print("\n🔍 Menerapkan filter domain knowledge...")
    df = df[df['ear'] <= 0.45]           # EAR maksimal realistis
    df = df[df['ear'] >= 0.02]           
    df = df[df['head_pose'] <= 0.35]     # Head pose maksimal
    df = df[df['mouth_ratio'] <= 0.15]   # 🔥 KRUSIAL: batasi mouth ratio
    
    # 2. Hapus mouth_ratio yang terlalu tinggi untuk kelas FOKUS (label 1)
    before_filter = len(df)
    df = df[~((df['label'] == 1) & (df['mouth_ratio'] > 0.10))]
    print(f"   Menghapus {before_filter - len(df)} baris dengan mouth_ratio > 0.10 pada kelas FOKUS")
    
    # 3. Hapus outlier statistik (IQR method) per kelas
    print("\n📊 Menghapus outlier statistik...")
    for label in [0, 1]:
        mask_label = df['label'] == label
        for col in ['ear', 'head_pose', 'mouth_ratio']:
            Q1 = df.loc[mask_label, col].quantile(0.25)
            Q3 = df.loc[mask_label, col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask_outlier = ~((df[col] < lower) | (df[col] > upper)) | (~mask_label)
            df = df[mask_outlier]
    
    print(f"   Data setelah filter outlier: {len(df)} baris")
    
    # 4. Balance dataset (sampling jika perlu)
    print("\n⚖️ Menyeimbangkan dataset...")
    count_0 = (df['label'] == 0).sum()
    count_1 = (df['label'] == 1).sum()
    
    if count_0 > count_1 * 1.5:
        # Downsample kelas 0
        df_0 = df[df['label'] == 0].sample(n=count_1, random_state=42)
        df_1 = df[df['label'] == 1]
        df = pd.concat([df_0, df_1])
        print(f"   Downsampled kelas 0 dari {count_0} ke {count_1}")
    elif count_1 > count_0 * 1.5:
        # Downsample kelas 1
        df_1 = df[df['label'] == 1].sample(n=count_0, random_state=42)
        df_0 = df[df['label'] == 0]
        df = pd.concat([df_0, df_1])
        print(f"   Downsampled kelas 1 dari {count_1} ke {count_0}")
    
    print(f"\n✅ Dataset bersih: {len(df)} baris ({len(df)/original_len*100:.1f}%)")
    print(f"   Label 0: {(df['label']==0).sum()}")
    print(f"   Label 1: {(df['label']==1).sum()}")
    
    # 5. Simpan dataset yang sudah dibersihkan
    df.to_csv('features_cleaned.csv', index=False)
    print("\n💾 Dataset bersih disimpan ke: features_cleaned.csv")
    
    # 6. Generate statistik baru
    print("\n📈 STATISTIK FITUR SETELAH CLEANING:")
    print("=" * 50)
    print(df[['ear', 'head_pose', 'mouth_ratio']].describe())
    
    print("\n📊 STATISTIK PER LABEL:")
    for label, name in [(0, 'TIDAK FOKUS'), (1, 'FOKUS')]:
        subset = df[df['label'] == label]
        print(f"\n{name} (n={len(subset)}):")
        print(f"  EAR    : mean={subset['ear'].mean():.3f}, max={subset['ear'].max():.3f}")
        print(f"  Head   : mean={subset['head_pose'].mean():.3f}, max={subset['head_pose'].max():.3f}")
        print(f"  Mouth  : mean={subset['mouth_ratio'].mean():.3f}, max={subset['mouth_ratio'].max():.3f}")
    
    return df

if __name__ == "__main__":
    clean_and_save_dataset()