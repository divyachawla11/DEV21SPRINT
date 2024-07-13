import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloading NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Loading the datasets
url = "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
df = pd.read_csv(url)

# Checking the column names
print(df.columns)
review_column = 'reviews.text'  

# Initializing Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Removing special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Converting to lowercase
    text = text.lower()
    
    # Tokenizing and removing stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Applying preprocessing
df['cleaned_text'] = df[review_column].apply(preprocess_text)

# Displaying the first few rows of cleaned text
print(df[[review_column, 'cleaned_text']].head())

# Defining the sentiment labeling function
def label_sentiment(rating):
    if rating >= 4:
        return 1
    elif rating <= 2:
        return 0
    else:
        return -1  # neutral reviews

# Labeling sentiments
rating_column = 'reviews.rating' 
df['sentiment'] = df[rating_column].apply(label_sentiment)

# Filtering out neutral reviews
df = df[df['sentiment'] != -1]

# Displaying the distribution of sentiments
print(df['sentiment'].value_counts())

# Train-test split
from sklearn.model_selection import train_test_split
X = df['cleaned_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape)
print(X_test_tfidf.shape)

# Model training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Model evaluation
from sklearn.metrics import accuracy_score, classification_report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
# Deployment with Streamlit (save as app.py)
import streamlit as st

st.title("Customer Review Sentiment Analysis")
review_text = st.text_area("Enter a customer review:")

if st.button("Analyze Sentiment"):
    cleaned_review = preprocess_text(review_text)
    review_tfidf = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_tfidf)
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"Sentiment: {sentiment}")
