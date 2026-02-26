📌 Movie Review Sentiment Analysis — Simple RNN (Deep Learning Project)

A complete end-to-end deep learning project for binary sentiment analysis on the IMDB Movie Review Dataset using a Simple Recurrent Neural Network (RNN).
The project includes:

Data loading & preprocessing

Word embeddings

Simple RNN model training

Sentiment prediction

Streamlit web application

Model deployment ready code

🚀 Project Overview

This project classifies movie reviews as Positive or Negative using deep learning.
It uses:

IMDB Movie Reviews Dataset

Word Index Encoding

Embedding Layer

SimpleRNN Layer

Binary Classification Layer

Streamlit Web UI for live prediction

The trained model achieves strong performance and can classify any custom user review.

🧠 Key Features

✔ End-to-end Deep Learning Pipeline
✔ Text Preprocessing & Tokenization
✔ Word Embedding Representation
✔ SimpleRNN-based Sentiment Model
✔ Real-Time Prediction with Streamlit
✔ Clean & Modular Code Structure
✔ Fully Reproducible Setup

📂 Project Structure
Movie-Review-Sentiment-Analysis/
│
├── SimpleRNN/
│   ├── main.py                # Streamlit web application
│   ├── prediction.ipynb       # Inference notebook
│   ├── embedding.ipynb        # Word embedding experiments
│   ├── simplernn.ipynb        # Training code using Simple RNN
│   ├── simple_rnn_imdb.h5     # Trained RNN model
│   └── requirements.txt       # Dependencies
│
└── README.md                  # Project documentation
🛠 Technologies Used
Library	Purpose
TensorFlow / Keras	Deep learning model
NumPy	Numeric processing
Streamlit	Web application UI
scikit-learn	Evaluation utilities
Matplotlib	Visualizations
IMDB Dataset	Sentiment data
📥 Dataset Information

We use the IMDB dataset, a popular dataset for sentiment analysis:

50,000 movie reviews

Binary sentiment: 0 = Negative, 1 = Positive

Pre-tokenized into integer sequences

Top 10,000 words vocabulary used

🧪 Model Architecture

The model architecture is simple yet effective:
Embedding (10000 → 128)
SimpleRNN (128 units, ReLU)
Dense (1, Sigmoid)
Loss Function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
▶️ How to Run the Streamlit App
1️⃣ Create & activate environment
conda create -n myenv python=3.10 -y
conda activate myenv
2️⃣ Install dependencies
pip install -r requirements.txt
pip install tensorflow==2.15.0
3️⃣ Run the Streamlit app
cd SimpleRNN
streamlit run main.py

App opens automatically at:

http://localhost:8501
🧩 Web App Features

Enter any movie review text

Model preprocesses & pads input

Predicts sentiment instantly

Shows prediction probability

Emoji-based output visual feedback

📊 Example Predictions

Input:

This movie was absolutely fantastic! Loved it.

Output:

Sentiment: Positive 😊
Prediction Score: 0.9453

Input:

The movie was boring and poorly directed.

Output:

Sentiment: Negative 😞
Prediction Score: 0.1247
🧱 Preprocessing Pipeline

✔ Lowercasing
✔ Tokenization
✔ Convert words → IMDB word index
✔ Replace out-of-vocabulary words with “unknown” token
✔ Fixed-length padding (500 tokens)

🧠 Why Simple RNN?

Easy to understand

Good for short sequences

Works as an introduction to sequence modeling

Helps understand fundamentals before LSTM/GRU/Transformers

📈 Future Enhancements

You can extend this project with:

🔥 LSTM / GRU model

🔥 Bidirectional RNN

🔥 Attention mechanism

🔥 Word2Vec / GloVe embeddings

🔥 Deploy on Render / HuggingFace

🔥 Replace with BERT for higher accuracy

❤️ Author

MSIVAPAPARAO13
📌 Passionate about Deep Learning, NLP, and AI Projects

GitHub Profile:
👉 https://github.com/MSIVAPAPARAO13

⭐ Show Your Support

If you found this project helpful:

✔ Star ⭐ the repository
✔ Share your feedback
✔ Fork the project and contribute
