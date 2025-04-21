import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Streamlit page config
st.set_page_config(page_title="Wine Quality KNN Classifier", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Options")
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type=["csv"])
k_selected = st.sidebar.slider("🔢 Select Number of Neighbors (K)", 1, 20, 5)
show_data = st.sidebar.checkbox("🔍 Show Raw Data")

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("WineQT1.csv")  # fallback
df.dropna(inplace=True)

# Convert quality to binary: Good (>=7), Bad (<7)
df["quality_binary"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

# Title
st.title("🍷 Wine Quality Predictor using KNN")
st.markdown("Upload a wine dataset, tune the K value, and predict wine quality with helpful visuals.")

# Show dataset
if show_data:
    st.subheader("📄 Raw Dataset")
    st.dataframe(df.head(20), use_container_width=True)

# Prepare data
X = df.drop(["quality", "quality_binary"], axis=1)
y = df["quality_binary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = KNeighborsClassifier(n_neighbors=k_selected)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# Accuracy display
st.subheader("📈 Model Performance")
st.success(f"✅ Accuracy with K = {k_selected}: **{acc:.2f}**")

# User input for prediction
st.subheader("🧪 Predict a New Sample")
st.markdown("Enter the feature values below to predict if the wine is **Good** or **Bad**.")

input_cols = st.columns(3)
user_input = []
for idx, col in enumerate(X.columns):
    with input_cols[idx % 3]:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        user_input.append(val)

if st.button("🚀 Predict Now"):
    input_scaled = scaler.transform([user_input])
    pred = model.predict(input_scaled)[0]
    result = "🍷 Good Quality Wine" if pred == 1 else "🍷 Bad Quality Wine"
    st.balloons()
    st.success(f"🎉 Prediction: **{result}**")

# Divider
st.markdown("---")

# Graphs section
st.subheader("📊 Visualizations")

col1, col2 = st.columns(2)

# Confusion Matrix
with col1:
    st.markdown("#### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBu", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# Bar graph for prediction distribution
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
