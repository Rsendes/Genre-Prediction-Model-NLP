import re
import csv
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import nltk
from nltk.corpus import stopwords
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_set = 'train_80.txt'
test_set = 'test_set_20.txt'
results = 'results_20.txt'

# Function to clean text (removes punctuation, converts to lowercase)
def clean_text(text):
    text = re.sub("\'", "", text)  # remove backslash-apostrophe
    text = re.sub("[^a-zA-Z]", " ", text)  # remove everything except alphabets
    text = ' '.join(text.split())  # remove extra whitespaces
    return text.lower()  # convert to lowercase

# Function to remove stopwords from text
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

### 1. Load and Process Training Data (from train.txt)
plots = []
genres = []
with open(train_set, 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        genres.append(row[2].split(','))  # Split genres into a list for multi-label classification
        plots.append(clean_text(remove_stopwords(row[4])))  # Clean and process the plot

# Initialize multi-label binarizer and transform genres
multilabel_binarizer = MultiLabelBinarizer()
y = multilabel_binarizer.fit_transform(genres)

# Initialize TF-IDF vectorizer and fit to plot summaries
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
x_tfidf = tfidf_vectorizer.fit_transform(plots)

# Train the model (Logistic Regression with One-vs-Rest)
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(x_tfidf, y)

### 2. Load and Process Test Data (from test_set.txt, includes true genres)
test_plots = []
true_genres = []
with open(test_set, 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        true_genres.append(row[2].split(','))  # True genres for evaluation
        test_plots.append(clean_text(remove_stopwords(row[4])))  # Clean and process the plot

# Transform the test plots into TF-IDF features using the trained TF-IDF vectorizer
x_test_tfidf = tfidf_vectorizer.transform(test_plots)

# Convert true genres into binary format (similar to training)
y_true = multilabel_binarizer.transform(true_genres)

### 3. Predict Genres for Test Data (single genre per plot)

# Predict the probabilities on the test dataset
y_test_pred_prob = clf.predict_proba(x_test_tfidf)

# Get the genre with the highest probability for each plot
y_pred_single = y_test_pred_prob.argmax(axis=1)

# Convert the integer predictions back to genre labels
predicted_genres = multilabel_binarizer.classes_[y_pred_single]

# Convert true genres into single label (for comparison, take the first true genre in the list)
true_single_genres = [genre[0] for genre in true_genres]

### 4. Save the Predictions and True Genres to a File

# Write the true and predicted genres to a text file
with open(results, 'w') as f:
    for true, pred in zip(true_single_genres, predicted_genres):
        f.write(f"True: {true}\n")
        f.write(f"Predicted: {pred}\n")
        f.write("\n")

### 5. Evaluate the Model with F1 Score and Confusion Matrix

# Calculate F1 score (for single genre prediction)
f1 = f1_score(true_single_genres, predicted_genres, average='macro')
f1_weighted = f1_score(true_single_genres, predicted_genres, average='weighted')
accuracy = accuracy_score(true_single_genres, predicted_genres)

# Generate classification report (with precision, recall, f1-score per genre)
report = classification_report(true_single_genres, predicted_genres, target_names=multilabel_binarizer.classes_)

# Calculate confusion matrix
cm = confusion_matrix(true_single_genres, predicted_genres, labels=multilabel_binarizer.classes_)

# Append evaluation results to the same file
with open(results, 'a') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Macro-Average F1 Score: {f1}\n")
    f.write(f"Weighted F1 Score: {f1_weighted}\n")
    f.write("\nClassification Report:\n")
    f.write(report)

print("Predictions, F1 scores, and evaluation metrics saved to " + results)

"""
### 6. Visualize Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=multilabel_binarizer.classes_, yticklabels=multilabel_binarizer.classes_)
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')
plt.savefig('confusion_matrix.png')
"""
