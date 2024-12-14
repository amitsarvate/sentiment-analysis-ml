import torch
from new_models import FeedforwardNeuralNetwork, ConvolutionalNeuralNetwork, GatedRecurrentUnit
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

# Set models to evaluation mode
modelFNN.eval()
modelCNN.eval()
modelGRU.eval()


def preprocess_review(review, vocab, max_len):
    tokens = [vocab.get(word, vocab['<UNK>']) for word in review.split()]
    padded_tokens = pad_sequence([torch.tensor(tokens[:max_len])], batch_first=True, padding_value=0)
    return padded_tokens


def predict_sentiment(review, model, vocab, max_len):
    # Preprocess the review
    processed_review = preprocess_review(review, vocab, max_len)
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model(processed_review))
        sentiment = "positive" if output.item() > 0.5 else "negative"
    return sentiment



if __name__ == "__main__":
    # Load the vocabulary and model
    max_len = 100
    

    print("Sentiment Analysis App")
    print("Type 'exit' to quit the app.")

    while True:
        user_review = input("\nEnter a movie review: ")
        if user_review.lower() == "exit":
            print("Goodbye!")
            break

        sentiment = predict_sentiment(user_review, modelGRU, vocab, max_len)
        print(f"The predicted sentiment is: {sentiment}")