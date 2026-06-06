#!/bin/bash

echo "=========================================="
echo "FocusWebCam - Fixed Version Setup"
echo "=========================================="

# Step 1: Clean dataset
echo ""
echo "📊 Step 1: Cleaning dataset..."
python3 clean_dataset.py

# Step 2: Train new model
echo ""
echo "🤖 Step 2: Training improved model..."
python3 train_model.py

# Step 3: Run Streamlit app
echo ""
echo "🚀 Step 3: Starting Streamlit app..."
echo "   Streamlit akan terbuka di http://localhost:8501"
echo "   Tekan Ctrl+C untuk menghentikan"
echo ""

streamlit run app.py