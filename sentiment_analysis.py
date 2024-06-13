import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv('/Users/linyun/Downloads/IMDB Dataset.csv')

# Function to clean the text
def clean_text(text):
    text = re.sub(r'<br />', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading and trailing whitespaces
    tokens = text.split()  # Split text into tokens
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Apply text cleaning to the dataset
df['review'] = df['review'].apply(clean_text)

# Check for missing values
print(df.isnull().sum())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Convert labels to binary (positive: 1, negative: 0)
y_train = y_train.apply(lambda x: 1 if x == 'positive' else 0)
y_test = y_test.apply(lambda x: 1 if x == 'positive' else 0)

# Define the classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(),
    'KNeighbors': KNeighborsClassifier(),
    'NaiveBayes': MultinomialNB()
}

# Create pipelines for each classifier
pipelines = {}
for key, classifier in classifiers.items():
    pipelines[key] = Pipeline([
        ('vect', CountVectorizer(stop_words='english', max_df=0.7)),
        ('clf', classifier)
    ])

# Train and evaluate each model
for key, pipeline in pipelines.items():
    print(f"Training {key} model...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f'{key} Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Save the model and vectorizer
    with open(f'sentiment_model_{key}.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)

# Function to predict sentiment of a user-provided review using a specific model
def predict_sentiment(review, model_name):
    review = clean_text(review)  # Clean the input review
    with open(f'sentiment_model_{model_name}.pkl', 'rb') as model_file:
        pipeline = pickle.load(model_file)
    prediction = pipeline.predict([review])
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Example usage
model_name = input("Enter the model to use (RandomForest, KNeighbors, NaiveBayes): ")
user_review = input("Enter a movie review: ")
print(f'The sentiment of the review using {model_name} model is: {predict_sentiment(user_review, model_name)}')


