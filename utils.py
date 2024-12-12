import os
import pickle
import string
import sqlite3

import nltk
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from pandas import DataFrame
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('punkt')


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



def build_index(data: DataFrame, column: str, stopwords: bool, settings: dict = {}):
    # Get all rows from the specified column
    questions = data.loc[:, column]

    # Apply preprocessing to each question in the corpus
    # do not remove stops words for intent matrix
    questions = questions.apply(lambda q: preprocess(q, remove_stopwords=stopwords))

    # Create a TF-IDF term-document matrix
    vectorizer = TfidfVectorizer(**settings)
    tfidf_matrix = vectorizer.fit_transform(questions)

    return tfidf_matrix, vectorizer


def load_model(corpus: DataFrame, column_name: str, remove_stopwords: bool,
               settings, model_name: str):
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


def initialize_database():
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()

    # Create the Name table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Name (
            Username VARCHAR(255) PRIMARY KEY
        )
    ''')

    # Create the Tickets table with a foreign key to the Name table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Tickets (
            TicketID INTEGER PRIMARY KEY AUTOINCREMENT,
            Username VARCHAR(255),
            DepartureCity VARCHAR(255),
            DestinationCity VARCHAR(255),
            TravelDateTime DATETIME,
            FOREIGN KEY (Username) REFERENCES Name(Username)
        )
    ''')

    connection.commit()
    connection.close()


def save_name(name):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()

    cursor.execute('SELECT Username FROM Name WHERE Username=?', (name,))
    result = cursor.fetchone()

    if result is None:
        cursor.execute('INSERT INTO Name (Username) VALUES (?)', (name,))
        connection.commit()
        print(f"Digital Conductor: Hello {name}, I will make sure to remember you next time.")
    else:
        print(f"Digital Conductor: Welcome back {name}! How can I help you today?")

    connection.close()


def save_ticket(username, departure_city, destination_city, travel_datetime):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()

    # Convert datetime object to string in ISO format
    travel_datetime = travel_datetime.strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute('''
        INSERT INTO Tickets (Username, DepartureCity, DestinationCity, TravelDateTime)
        VALUES (?, ?, ?, ?)''', (username, departure_city, destination_city, travel_datetime))

    ticket_id = cursor.lastrowid

    connection.commit()
    connection.close()
    print(f"Digital Conductor: Your reference number is {ticket_id}")


def get_tickets(username):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()

    cursor.execute('''
        SELECT DepartureCity, DestinationCity, TravelDateTime
        FROM Tickets
        WHERE Username=?''', (username,))
    tickets = cursor.fetchall()
    connection.close()
    return tickets


def get_ticket(reference_number):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()

    cursor.execute('''
        SELECT DepartureCity, DestinationCity, TravelDateTime
        FROM Tickets
        WHERE TicketID=?''', (reference_number,))
    ticket = cursor.fetchone()
    connection.close()
    return ticket
