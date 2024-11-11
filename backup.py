import pandas as pd
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import nltk
import numpy as np

# Configure display option for pandas
pd.set_option('display.max_colwidth', 300)

# Read and process movie data
plots = []
with open("train.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)

# Initialize lists for columns
title, origin, genre, director, plot = [], [], [], [], []

# Extract relevant information into lists
for i in tqdm(plots):
    title.append(i[0])
    origin.append(i[1])
    genre.append(i[2])
    director.append(i[3])
    plot.append(i[4])

# Create dataframe from lists
movies = pd.DataFrame({'title': title, 'origin': origin, 'genre': genre, 'director': director, 'plot': plot})

# Split genres into a list for multi-label classification
movies['genre'] = movies['genre'].apply(lambda x: x.split(','))

# Function to clean text (removes punctuation, converts to lowercase)
def clean_text(text):
    text = re.sub("\'", "", text)  # remove backslash-apostrophe
    text = re.sub("[^a-zA-Z]", " ", text)  # remove everything except alphabets
    text = ' '.join(text.split())  # remove extra whitespaces
    return text.lower()  # convert to lowercase

# Clean the plot summaries
movies['clean_plot'] = movies['plot'].apply(clean_text)

# Download stopwords
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from text
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# Apply stopword removal to the cleaned plot summaries
movies['clean_plot'] = movies['clean_plot'].apply(remove_stopwords)

# Initialize multi-label binarizer and transform genres
multilabel_binarizer = MultiLabelBinarizer()
y = multilabel_binarizer.fit_transform(movies['genre'])

# Initialize TF-IDF vectorizer and fit to plot summaries
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# Convert the plot data to TF-IDF features
x_tfidf = tfidf_vectorizer.fit_transform(movies['clean_plot'])

# Number of folds for cross-validation
k = 5

# Initialize the k-fold cross-validation setup
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=9)

# Hyperparameter: range of thresholds to try
threshold_values = [0.2, 0.3, 0.4, 0.5]

# Dictionary to store results for each threshold
results = {t: [] for t in threshold_values}

# Use StratifiedKFold to split the data
for train_index, val_index in kf.split(x_tfidf, y.argmax(axis=1)):  # Using the argmax for stratification
    # Split data into train and validation sets
    xtrain_tfidf, xval_tfidf = x_tfidf[train_index], x_tfidf[val_index]
    ytrain, yval = y[train_index], y[val_index]
    
    # Initialize and train the model (Logistic Regression with One-vs-Rest)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(xtrain_tfidf, ytrain)

    # Get predicted probabilities for validation set
    y_pred_prob = clf.predict_proba(xval_tfidf)

    # Try different threshold values
    for t in threshold_values:
        # Apply threshold to convert probabilities into binary predictions
        y_pred = (y_pred_prob >= t).astype(int)

        # Evaluate performance using F1 score (micro)
        f1 = f1_score(yval, y_pred, average="micro")

        # Store the F1 score for this fold
        results[t].append(f1)

# Calculate the average F1 score across folds for each threshold
average_f1_scores = {t: np.mean(f1s) for t, f1s in results.items()}

# Find the threshold with the best average F1 score
best_threshold = max(average_f1_scores, key=average_f1_scores.get)
print(f"Best threshold: {best_threshold}")
print(f"F1 scores for different thresholds: {average_f1_scores}")

# Once the best threshold is found, we train the model on the full dataset
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(x_tfidf, y)

# Predict the probabilities on the full dataset
y_pred_prob = clf.predict_proba(x_tfidf)

# Get the genre with the highest probability for each movie plot
single_genre_predictions = y_pred_prob.argmax(axis=1)

# Convert the integer predictions back to genre labels (one per plot)
single_genre_labels = multilabel_binarizer.classes_[single_genre_predictions]

# Write the single predicted genre per plot to a text file
with open('results.txt', 'w') as f:
    for genre in single_genre_labels:
        f.write(f"{genre}\n")

print("Predictions saved to results.txt")

# Load the test file (test_no_labels.txt)
test_plots = []

with open("test_no_labels.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        test_plots.append(row)

# Initialize lists for columns in the test set
test_title, test_origin, test_director, test_plot = [], [], [], []

# Extract relevant information from the test file
for i in tqdm(test_plots):
    test_title.append(i[0])
    test_origin.append(i[1])
    test_director.append(i[2])
    test_plot.append(i[3])

# Create a dataframe for the test data
test_movies = pd.DataFrame({'title': test_title, 'origin': test_origin, 'director': test_director, 'plot': test_plot})

# Clean the test plots in the same way as the training data
test_movies['clean_plot'] = test_movies['plot'].apply(clean_text)
test_movies['clean_plot'] = test_movies['clean_plot'].apply(remove_stopwords)

# Transform the test plots into TF-IDF features using the trained TF-IDF vectorizer
x_test_tfidf = tfidf_vectorizer.transform(test_movies['clean_plot'])

# Predict the probabilities on the test dataset
y_test_pred_prob = clf.predict_proba(x_test_tfidf)

# Get the genre with the highest probability for each movie plot in the test set
single_test_genre_predictions = y_test_pred_prob.argmax(axis=1)

# Convert the integer predictions back to genre labels (one per plot)
single_test_genre_labels = multilabel_binarizer.classes_[single_test_genre_predictions]

# Write the single predicted genre per plot to a text file (results_test.txt)
with open('results_test.txt', 'w') as f:
    for genre in single_test_genre_labels:
        f.write(f"{genre}\n")

print("Predictions saved to results_test.txt")