import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def download_and_extract_dataset():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    dataset_path = 'smsspamcollection.zip'
    dataset_folder = 'smsspamcollection'

    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, dataset_path)

    if not os.path.exists(dataset_folder):
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)
    else:
        print("Dataset already extracted.")

    data_file = os.path.join(dataset_folder, 'SMSSpamCollection')
    return data_file

def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    print(f"Dataset loaded with {len(df)} entries.")
    return df

def preprocess_and_split(df):
    print("Preprocessing data...")

    # Encode labels (ham=0, spam=1)
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'], test_size=0.25, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("Training the model...")

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    pipeline.fit(X_train, y_train)
    print("Model trained successfully.")
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    print("Evaluating the model...")

    predicted = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, predicted, target_names=['Ham', 'Spam']))

def main():
    data_file = download_and_extract_dataset()
    df = load_data(data_file)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
