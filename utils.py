import os
import pickle
import string

import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from pandas import DataFrame
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(raw_text, remove_stopwords):
    lemmatizer = WordNetLemmatizer()
    raw_text = raw_text.lower()
    # Modify string.punctuation
    string.punctuation = string.punctuation + "’" + "-" + "‘" + "-"
    string.punctuation = string.punctuation.replace('.', '')
    # Use the modified string.punctuation to remove punctuation from raw_text
    raw_text = raw_text.translate(str.maketrans('', '', string.punctuation))

    filtered_tokens = nltk.word_tokenize(raw_text)
    processed_tokens = []

    if remove_stopwords:
        filtered_tokens = [word for word in filtered_tokens if word not in stopwords.words('english')]

    posmap = {
        'ADJ': 'a',
        'ADV': 'r',
        'NOUN': 'n',
        'VERB': 'v',
    }

    post = nltk.pos_tag(filtered_tokens, tagset='universal')
    for token in post:
        word = token[0]
        tag = token[1]
        if tag in posmap.keys():
            processed_tokens.append(lemmatizer.lemmatize(word, posmap[tag]))
        else:
            processed_tokens.append(lemmatizer.lemmatize(word))

    # return a preprocessed string
    # print(processed_tokens)
    return " ".join(processed_tokens)


def query_similarity(query, tfidf_matrix, vectorizer):
    # Transform the query to match the TF-IDF representation
    query_vector = vectorizer.transform([query])

    # Convert to dense arrays for the dot product
    query_array = query_vector.toarray().flatten()
    doc_arrays = tfidf_matrix.toarray()

    # Calculate similarities using vectorized operations
    similarity_scores = []
    for doc_vector in doc_arrays:
        # Check if either vector is zero magnitude to avoid divide-by-zero error
        if np.linalg.norm(query_array) == 0 or np.linalg.norm(doc_vector) == 0:
            similarity = 0  # Assign a similarity score of 0 if either vector has zero magnitude
        else:
            similarity = 1 - cosine(query_array, doc_vector)
        similarity_scores.append(similarity)

    # Create results DataFrame
    results = pd.DataFrame({
        'Document': [f'Q{i + 1}' for i in range(tfidf_matrix.shape[0])],
        'Cosine Similarity': similarity_scores
    })

    return results.sort_values(by='Cosine Similarity', ascending=False)


def build_index(data:DataFrame, column:str, stopwords:bool, settings:dict=None):
    questions = data.loc[:, column]

    # Apply preprocessing to each question in the corpus
    # do not remove stops words for intent matrix
    questions = questions.apply(lambda q: preprocess(q, remove_stopwords=stopwords))

    # Create a TF-IDF term-document matrix
    if settings is None:
        settings = {}

    vectorizer = TfidfVectorizer(**settings)
    # vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(questions)

    return tfidf_matrix, vectorizer


def load_model(corpus:DataFrame, column_name:str, remove_stopwords:bool,
               settings, model_name:str):
    # Directory to save the pickle files
    pickle_dir = "models"
    os.makedirs(pickle_dir, exist_ok=True)

    # File paths for the model
    matrix_file = os.path.join(pickle_dir, f"{model_name}_matrix.pkl")
    vectorizer_file = os.path.join(pickle_dir, f"{model_name}_vectorizer.pkl")

    # Load or build the model
    if os.path.exists(matrix_file) and os.path.exists(vectorizer_file):
        print(f"Found {model_name} model")
        with open(matrix_file, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        print(f"Building {model_name} model")
        tfidf_matrix, vectorizer = build_index(corpus, column_name, remove_stopwords, settings)
        with open(matrix_file, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizer, f)

    return tfidf_matrix, vectorizer