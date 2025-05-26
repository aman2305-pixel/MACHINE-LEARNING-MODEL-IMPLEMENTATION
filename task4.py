import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encode labels
    return df

def preprocess_data(df):
    X = df['message']
    y = df['label']
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    return X_vec, y, vectorizer

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def main():
    df = load_data()
    X_vec, y, vectorizer = preprocess_data(df)
    model, X_test, y_test = train_model(X_vec, y)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
