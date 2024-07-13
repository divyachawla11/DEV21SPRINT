**Customer Review Sentiment Analysis:-**

**_1. Project Overview:_**
Our project aims to build a sentiment analysis system that classifies customer reviews into three sentiment categories: Positive, Negative, and Neutral. The system utilizes Natural Language Processing (NLP) techniques and machine learning algorithms to analyze and categorize sentiments from textual data.

_**2. Features:-**_

**Text Preprocessing:** Clean and preprocess the text data by removing special characters, digits, and stopwords, and by lemmatizing the words.

**Vectorization:** Convert text into numerical features using TF-IDF vectorization.

**Sentiment Classification:** Classify sentiments using Logistic Regression.

**Interactive Web App:** We have used _Streamlit_ to provide an easy-to-use web interface for analyzing customer reviews.


_**3. Project Structure:-**_

**sentiment_analysis.py:** Streamlit app for deploying the sentiment analysis model.

**sentiment_model.pkl:** Saved sentiment analysis model.

**tfidf_vectorizer.pkl:** Saved TF-IDF vectorizer.

**README.md:** Project overview and instructions (this file).


_**4. Usage:-**_

**Clone the repository:**

git clone https://github.com/divyachawla11/DEV21SPRINT.git

cd DEV21SPRINT

**Create a virtual environment and activate it:**

python -m venv env

source env/bin/activate  # On Windows, use `env\Scripts\activate`

**Download NLTK data:**

import nltk

nltk.download('stopwords')

nltk.download('wordnet')

**Running the Streamlit App:-**

Run the Streamlit app to interact with the sentiment analysis model:

streamlit run sentiment_analysis.py

Enter a customer review in the text area and click "Analyze Sentiment" to see the predicted sentiment category.

_**5. File Descriptions:-**_

**sentiment_analysis.py:** Contains the Streamlit app code for deploying the sentiment analysis model.

**README.md:** Provides an overview and instructions for the project.

_**6. Acknowledgments:-**_

This project uses data and tools from the Natural Language Toolkit (NLTK) and Scikit-learn libraries.

Special thanks to the open-source community for providing valuable resources and support.
