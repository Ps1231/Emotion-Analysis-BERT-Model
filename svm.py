from sklearn.svm import LinearSVC
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import re
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import nltk
nltk.download('punkt')

# Function for text preprocessing


def preprocess_and_tokenize(data):
    # Remove html markup
    data = re.sub("(<.*?>)", "", data)
    # Remove urls
    data = re.sub(r'http\S+', '', data)
    # Remove hashtags and @names
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    # Remove punctuation and non-ascii digits
    data = re.sub("(\\W|\\d)", " ", data)
    # Remove whitespace
    data = data.strip()
    # Tokenization with nltk
    data = word_tokenize(data)
    # Stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]
    return stem_data


# Load data
df_train = pd.read_csv('/content/data_train.csv')
df_test = pd.read_csv('/content/data_test.csv')

X_train = df_train.Text
X_test = df_test.Text

y_train = df_train.Emotion
y_test = df_test.Emotion

# Concatenate data for TF-IDF vectorization
data = pd.concat([df_train, df_test])

# TFIDF, unigrams and bigrams
vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize,
                       sublinear_tf=True, norm='l2', ngram_range=(1, 2))

# Fit on complete corpus
vect.fit_transform(data.Text)

# Transform training and testing datasets to vectors
X_train_vect = vect.transform(X_train)
X_test_vect = vect.transform(X_test)

# Create and train the model
svc = LinearSVC(tol=1e-05)
svc.fit(X_train_vect, y_train)

# Create pipeline with TF-IDF vectorizer and LinearSVC model
svm_model = Pipeline([
    ('tfidf', vect),
    ('clf', svc),
])

# Save the model
filename = '/content/tfidf_svm.sav'
pickle.dump(svm_model, open(filename, 'wb'))

# Load the model from file
loaded_model = pickle.load(open(filename, 'rb'))

# Function to predict emotion using the loaded model


def predict_emotion(message, model):
    if model:
        return model.predict([message])
    else:
        return None


# Example usage
message = 'today is monday '
predicted_emotion = predict_emotion(message, loaded_model)
print('Predicted emotion:', predicted_emotion)
