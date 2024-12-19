# Sentiment Analysis Application

This project provides a sentiment analysis tool that predicts whether a movie review has a positive or negative sentiment. It employs machine learning models including Feedforward Neural Networks (FNN), Convolutional Neural Networks (CNN), and Gated Recurrent Units (GRU). 

## Description
The sentiment analysis application uses pre-trained models to classify the sentiment of user-inputted text. It is designed to:

* Preprocess textual data to make it suitable for sentiment analysis.

* Use machine learning models to classify sentiments as "positive" or "negative".

* Be interactive, allowing users to input their own reviews and get immediate predictions.

## How to Run Project

### Prerequisites
Ensure you have Python installed on your system along with the required dependencies listed in the requirements.txt file.

### Local Setup
 
1. Clone this repository to your local machine
2. Navigate to project directory
3. Install required dependencies
   `pip install -r requirements.txt`
5. Ensure the necessary model files (fnn_sentiment_model.pth, cnn_sentiment_model.pth, gru_sentiment_model.pth) and vocab.json are present in the project directory.
6. Run the review.py script
   `python review.py`
7. Follow prompts to input a review and get predictions

## Dependencies
The project uses the following libraries (see requirements.txt):

* streamlit

*torch

* torchvision

*numpy

* pandas

* requests

* aiohttp

* scikit-learn

* matplotlib

* psutil

## Data
The project uses a dataset named IMDB Dataset.csv containing movie reviews for training and evaluation.
