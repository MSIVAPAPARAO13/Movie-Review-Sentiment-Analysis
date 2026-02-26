# -----------------------------------------------------------
# Step 1: Import Libraries
# -----------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import streamlit as st

# -----------------------------------------------------------
# Step 2: Load IMDB word index and reverse mapping
# -----------------------------------------------------------
max_features = 10000  # MUST match training
word_index = imdb.get_word_index()

reverse_word_index = {value: key for key, value in word_index.items()}

# -----------------------------------------------------------
# Step 3: Load the trained model
# -----------------------------------------------------------
model = load_model("simple_rnn_imdb.h5")

# -----------------------------------------------------------
# Helper Function: Decode review (optional)
# -----------------------------------------------------------
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# -----------------------------------------------------------
# Helper Function: Preprocess User Input
# ----- FIXED TO AVOID embedding index errors -----
# -----------------------------------------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        index = word_index.get(word, 2)  # unknown = 2
        if index >= max_features:
            index = 2  # force unknown for out-of-range
        encoded_review.append(index + 3)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative.")

# User Input Box
user_input = st.text_area("Movie Review", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please type a review first.")
    else:
        # Preprocess text
        processed_input = preprocess_text(user_input)

        # Predict Sentiment
        prediction = model.predict(processed_input)
        score = float(prediction[0][0])

        sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"

        # Display
        st.subheader("Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {score:.4f}")

else:
    st.info("Enter a review and click Classify.")