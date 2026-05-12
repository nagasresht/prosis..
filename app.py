import streamlit as st
from PIL import Image
from predict import predict

st.title("Early Psoriasis Detection System")
st.write("Binary classification: Psoriasis vs Non-Psoriasis")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        label, confidence = predict(image)

        st.success(f"Prediction: {label}")
        st.write(f"Confidence: {confidence}%")