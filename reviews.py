import re
import csv
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import nltk
from nltk.corpus import stopwords

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
with open("train.txt", 'r') as f:
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

### 2. Load and Process Test Data (from test_no_labels.txt)
test_plots = []
with open("test_no_labels.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        test_plots.append(clean_text(remove_stopwords(row[3])))  # Clean and process the plot

# Transform the test plots into TF-IDF features using the trained TF-IDF vectorizer
x_test_tfidf = tfidf_vectorizer.transform(test_plots)

### 3. Predict Genres for Test Data and Save to results_test.txt

# Predict the probabilities on the test dataset
y_test_pred_prob = clf.predict_proba(x_test_tfidf)

# Get the genre with the highest probability for each movie plot in the test set
single_test_genre_predictions = y_test_pred_prob.argmax(axis=1)

# Convert the integer predictions back to genre labels (one per plot)
single_test_genre_labels = multilabel_binarizer.classes_[single_test_genre_predictions]

# Write the single predicted genre per plot to a text file (results_test.txt)
with open('results.txt', 'w') as f:
    for genre in single_test_genre_labels:
        f.write(f"{genre}\n")

print("Predictions saved to results.txt")
