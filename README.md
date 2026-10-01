# Movie Review Sentiment Analysis

An end-to-end Natural Language Processing and Deep Learning project for classifying movie reviews as **Positive** or **Negative** using a **Simple Recurrent Neural Network (Simple RNN)**.

**GitHub:** https://github.com/MSIVAPAPARAO13/Movie-Review-Sentiment-Analysis

---

## Project Overview

This project implements binary sentiment classification on the **IMDB Movie Review Dataset**.

The complete workflow includes:

```text
IMDB Reviews
     ↓
Word Index Encoding
     ↓
Sequence Padding
     ↓
Word Embedding
     ↓
SimpleRNN
     ↓
Sigmoid Output
     ↓
Positive / Negative Sentiment
```

A Streamlit application is included for interactive real-time prediction on custom movie reviews.

---

## Features

* Binary movie-review sentiment classification
* IMDB dataset integration
* 10,000-word vocabulary
* Word-index based text representation
* 128-dimensional word embeddings
* SimpleRNN sequence modeling
* 500-token fixed-length input sequences
* Prediction probability output
* Positive/Negative classification
* Interactive Streamlit UI
* Saved trained Keras model
* Reproducible development environment

---

## Dataset

The project uses the **IMDB Movie Reviews Dataset** provided through TensorFlow/Keras.

### Dataset Statistics

```text
Total Reviews : 50,000
Training      : 25,000
Testing       : 25,000
Vocabulary    : 10,000 words
Classes       : 2
```

### Target Classes

```text
0 → Negative
1 → Positive
```

The notebook verifies the training and testing shapes as:

```text
Training data shape: (25000,)
Testing data shape:  (25000,)
```

---

## Text Preprocessing

The project uses the IMDB word-index representation.

### Processing Steps

1. Convert text to lowercase.
2. Split the review into words.
3. Convert words into IMDB vocabulary indices.
4. Use the unknown-token representation for words outside the vocabulary.
5. Convert the review into an integer sequence.
6. Pad/truncate sequences to a fixed length of **500 tokens**.

The application implements additional protection against embedding-index errors by mapping vocabulary indices outside the configured 10,000-word range to the unknown token.

---

## Model Architecture

The trained model is stored as:

```text
simple_rnn_imdb.h5
```

### Architecture

```text
Input
  ↓
Embedding
10,000 vocabulary
128-dimensional vectors
  ↓
SimpleRNN
128 units
  ↓
Dense
1 neuron
Sigmoid
  ↓
Sentiment Probability
```

### Verified Model Summary

```text
Layer                  Output Shape          Parameters
-------------------------------------------------------
Embedding              (None, 500, 128)      1,280,000
SimpleRNN              (None, 128)              32,896
Dense                  (None, 1)                   129
-------------------------------------------------------
Total Parameters                              1,313,025
```

All **1,313,025 parameters** are trainable in the committed model summary.

---

## Training Configuration

The project uses:

```text
Loss Function : Binary Crossentropy
Optimizer     : Adam
Output        : Sigmoid
Task          : Binary Classification
```

---

## Prediction Logic

The output of the neural network is interpreted as a probability score.

```python
sentiment = "Positive" if score > 0.5 else "Negative"
```

Therefore:

```text
Score > 0.5  → Positive
Score ≤ 0.5  → Negative
```

---

## Streamlit Application

The interactive application is implemented in:

```text
main.py
```

Run the application using:

```bash
streamlit run main.py
```

The application provides:

* Movie review text area
* Classify button
* Input validation
* Sentiment result
* Prediction probability
* Emoji-based result display

---

## Example Prediction

The committed `prediction.ipynb` contains this example:

```text
This movie was fantastic! The acting was great and the plot was thrilling.
```

The stored notebook output is:

```text
Prediction Score: 0.42432793974876404
Sentiment: Negative
```

Because the score is below the configured threshold of `0.5`, the notebook classifies this example as Negative.

> Note: this is a single stored inference example and should not be interpreted as overall model accuracy.

---

## Project Structure

```text
Movie-Review-Sentiment-Analysis/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── README.md
├── main.py
├── embedding.ipynb
├── prediction.ipynb
├── simplernn.ipynb
├── simple_rnn_imdb.h5
└── requirements.txt
```

---

## File Description

### `main.py`

Streamlit application responsible for:

* Loading the IMDB word index
* Loading the trained model
* Preprocessing user reviews
* Padding sequences
* Performing sentiment prediction
* Displaying the result

### `simplernn.ipynb`

Training notebook containing the SimpleRNN development workflow.

### `embedding.ipynb`

Notebook used to inspect and explore the embedding representation.

### `prediction.ipynb`

Inference notebook containing:

* Model loading
* Review preprocessing
* Prediction function
* Example inference

### `simple_rnn_imdb.h5`

Serialized trained TensorFlow/Keras model.

### `requirements.txt`

Project dependencies.

---

## Requirements

The repository currently contains:

```text
tensorflow-cpu==2.12.0
streamlit
numpy
pandas
scikit-learn
tensorboard
matplotlib
scikeras
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MSIVAPAPARAO13/Movie-Review-Sentiment-Analysis.git
cd Movie-Review-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run main.py
```

By default, Streamlit runs on:

```text
http://localhost:8501
```

---

## Development Container

The repository also contains:

```text
.devcontainer/devcontainer.json
```

The configured development environment:

* Uses a Python Dev Container image
* Installs the project requirements
* Starts the Streamlit application
* Exposes port `8501`
* Opens the application preview automatically

---

## Technologies Used

| Technology         | Purpose               |
| ------------------ | --------------------- |
| Python             | Programming           |
| TensorFlow / Keras | Deep Learning         |
| SimpleRNN          | Sequence modeling     |
| IMDB Dataset       | Sentiment data        |
| NumPy              | Numerical operations  |
| Pandas             | Data processing       |
| Scikit-learn       | ML utilities          |
| Streamlit          | Web application       |
| Matplotlib         | Visualization         |
| TensorBoard        | Deep Learning tooling |

---

## Key Learning Outcomes

* Understanding NLP text representation
* Working with pre-tokenized datasets
* Building word embeddings
* Preparing sequential inputs
* Understanding recurrent neural networks
* Implementing SimpleRNN architectures
* Performing binary sentiment classification
* Saving and loading trained neural networks
* Building an interactive ML application with Streamlit

---

## Future Improvements

Potential extensions include:

* LSTM-based sentiment classification
* GRU-based architecture
* Bidirectional RNN
* Attention mechanism
* GloVe/Word2Vec embeddings
* Transformer-based sentiment models
* BERT-based classification
* Confusion matrix and ROC-AUC reporting
* Formal test-set metrics
* Cloud deployment

---

## Author

**Siva Paparao Medisetti**

GitHub: https://github.com/MSIVAPAPARAO13

Project Repository:

https://github.com/MSIVAPAPARAO13/Movie-Review-Sentiment-Analysis
