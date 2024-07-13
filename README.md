**Customer Review Sentiment Analysis:-**

**_Project Overview:_**
Our project aims to build a sentiment analysis system that classifies customer reviews into three sentiment categories: Positive, Negative, and Neutral. The system utilizes Natural Language Processing (NLP) techniques and machine learning algorithms to analyze and categorize sentiments from textual data.

_**Features:-**_

**Text Preprocessing:** Clean and preprocess the text data by removing special characters, digits, and stopwords, and by lemmatizing the words.

**Vectorization:** Convert text into numerical features using TF-IDF vectorization.

**Sentiment Classification:** Classify sentiments using Logistic Regression.

**Interactive Web App:** We have used _Streamlit_ to provide an easy-to-use web interface for analyzing customer reviews.

_**Project Structure:-**_

**DEVSPRINT code file.py:** Streamlit app for deploying the sentiment analysis model.

**sentiment_model.pkl:** Saved sentiment analysis model.

**tfidf_vectorizer.pkl:** Saved TF-IDF vectorizer.

**requirements.txt:** List of dependencies required to run the project.

**README.md:** Project overview and instructions (this file).

_**Installation:-**_

**Clone the repository:**

git clone https://github.com/divyachawla11/DEV21SPRINT.git

cd DEV21SPRINT

**Create a virtual environment and activate it:**

python -m venv env

source env/bin/activate  # On Windows, use `env\Scripts\activate`

**Install the required packages:**

pip install -r requirements.txt

**Download NLTK data:**

import nltk

nltk.download('stopwords')

nltk.download('wordnet')


_**Running the Streamlit App:-**_

Run the Streamlit app to interact with the sentiment analysis model:

streamlit run DEVSPRINT code file.py

Enter a customer review in the text area and click "Analyze Sentiment" to see the predicted sentiment category.

_**File Descriptions:-**_

**DEVSPRINT code file.py:** Contains the Streamlit app code for deploying the sentiment analysis model.

**requirements.txt:** Lists the required Python packages for the project.
**README.md:** Provides an overview and instructions for the project.


_**Acknowledgments:-**_

This project uses data and tools from the Natural Language Toolkit (NLTK) and Scikit-learn libraries.

Special thanks to the open-source community for providing valuable resources and support.
