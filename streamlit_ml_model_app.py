import streamlit as st
import torch
import torch.nn.functional as F
from models import FeedforwardNeuralNetwork, ConvolutionalNeuralNetwork, GatedRecurrentUnit
from torch.nn.utils.rnn import pad_sequence
import json


# Load the model
with open("vocab.json", "r") as f:
    vocab = json.load(f)
vocab_size = len(vocab)  # Use the same vocab size as during training
embed_dim = 100
hidden_dim = 128
output_dim = 1
num_filters = 128
kernel_sizes = [3, 4, 5]
num_layers = 1
max_len = 100

modelFNN = FeedforwardNeuralNetwork(vocab_size, embed_dim, hidden_dim, output_dim, max_len)
modelCNN = ConvolutionalNeuralNetwork(vocab_size, embed_dim, num_filters, kernel_sizes, output_dim)
modelGRU = GatedRecurrentUnit(vocab_size, embed_dim, hidden_dim, num_layers, output_dim)

modelFNN.load_state_dict(torch.load("fnn_sentiment_model.pth"))
modelCNN.load_state_dict(torch.load("cnn_sentiment_model.pth"))
modelGRU.load_state_dict(torch.load("gru_sentiment_model.pth"))

modelFNN.eval()
modelCNN.eval()
modelGRU.eval()


def preprocess_review(review, vocab, max_len):
    tokens = [vocab.get(word, vocab['<UNK>']) for word in review.split()]
    token_tensor = torch.tensor(tokens[:max_len], dtype=torch.long)  # Explicitly set dtype to torch.long (int64)
    if len(token_tensor) < max_len:
        padded_tokens = F.pad(token_tensor, (0, max_len - token_tensor.size(0)), value=0)  # Ensure fixed size
    else:
        padded_tokens = token_tensor
    padded_tokens = padded_tokens.unsqueeze(0)  # Add batch dimension
    return padded_tokens

def predict_sentiment(review, model, vocab, max_len):
    # Preprocess the review
    processed_review = preprocess_review(review, vocab, max_len)
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(processed_review))
        sentiment = "positive" if output.item() > 0.5 else "negative"
    return sentiment


st.title('Movie Review Sentiment Prediction')
user_review = st.text_input("Please enter your movie review: ")

if user_review:  # Ensure input exists before predicting
    sentiment = predict_sentiment(user_review, modelGRU, vocab, max_len)
    st.text(f"The predicted sentiment is: {sentiment}")
