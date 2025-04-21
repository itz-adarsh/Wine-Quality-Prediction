# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load model, scaler, and feature names
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Streamlit config
st.set_page_config(page_title="Wine Quality KNN Classifier", layout="wide")
st.sidebar.title("⚙️ Options")

# File uploader
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type=["csv"])
show_data = st.sidebar.checkbox("🔍 Show Raw Data")

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("WineQT1.csv")
df.dropna(inplace=True)
df["quality_binary"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

st.title("🍷 Wine Quality Classifier using KNN")

if show_data:
    st.subheader("📄 Raw Dataset")
    st.dataframe(df.head(20), use_container_width=True)

# Prepare test data
X = df[feature_names]
y = df["quality_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)

# Predict and accuracy
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
st.subheader("📈 Model Performance")
st.success(f"✅ Accuracy: **{acc:.2f}**")

# Prediction input
st.subheader("🧪 Predict a New Sample")
input_cols = st.columns(3)
user_input = []
for idx, col in enumerate(feature_names):
    with input_cols[idx % 3]:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        user_input.append(val)

if st.button("🚀 Predict Now"):
    input_scaled = scaler.transform([user_input])
    pred = model.predict(input_scaled)[0]
    result = "🍷 Good Quality Wine" if pred == 1 else "🍷 Bad Quality Wine"
    st.balloons()
    st.success(f"🎉 Prediction: **{result}**")

# Visualizations
st.markdown("---")
st.subheader("📊 Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBu", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

with col2:
    st.markdown("#### Prediction Count (Bar Chart)")
    counts = np.bincount(y_pred)
    labels = ['Bad', 'Good']
    fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
    ax_bar.bar(labels, counts, color=['#FF9999', '#99CC99'])
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Predicted Class Distribution")
    st.pyplot(fig_bar)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Adarsh using Streamlit")
