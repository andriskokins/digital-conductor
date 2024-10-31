import sqlite3

import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np


def preprocess(raw_text, remove_stopwords):
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = nltk.word_tokenize(raw_text.lower())
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


def read_csv(filepath):
    df = pd.read_csv(filepath, index_col='QuestionID')
    return df


def build_index(data, column):
    questions = data.loc[:, column]

    # Apply preprocessing to each question in the Series
    questions = questions.apply(lambda q: preprocess(q, remove_stopwords=True))

    # Create a TF-IDF term-document matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)  # Keep as sparse matrix

    # Get terms for reference
    terms = vectorizer.get_feature_names_out()

    # Convert to DataFrame for visualization (optional)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=terms,
        index=[f'Q{i + 1}' for i in range(len(questions))]
    )

    return tfidf_matrix, vectorizer  # Return sparse matrix instead of DataFrame


def query_similarity(query, tfidf_matrix, vectorizer):
    # Transform the query to match the TF-IDF representation
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity using matrix operations
    # Convert to dense arrays for the dot product
    query_array = query_vector.toarray().flatten()
    doc_arrays = tfidf_matrix.toarray()

    # Calculate similarities using vectorized operations
    similarity_scores = []
    for doc_vector in doc_arrays:
        similarity = 1 - cosine(query_array, doc_vector)
        similarity_scores.append(similarity)

    # Create results DataFrame
    results = pd.DataFrame({
        'Document': [f'Q{i + 1}' for i in range(tfidf_matrix.shape[0])],
        'Cosine Similarity': similarity_scores
    })

    return results.sort_values(by='Cosine Similarity', ascending=False)

intent_patterns = {
    'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
    'identity': ['what is my name', 'who am i'],
    'question': ['what is', 'what are', 'how do', 'how', 'how are', 'how much', 'what'],
    'small_talk': ['how are you', "what's up", "how's it going", 'how are things'],
    'discoverability': ['what can you do', 'help', 'what are your capabilities', 'what do you know']
}

def intent_match(query):
    # Check if the query matches any intent pattern
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if pattern in query.lower():
                return intent
    return None


def identity(name):
    connection = sqlite3.connect('names.db')
    cursor = connection.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS Name
                            (Username VARCHAR(255))''')

    cursor.execute('''SELECT username FROM Name WHERE username=?''', (name,))
    result = cursor.fetchone()

    if result is None:
        cursor.execute('''INSERT INTO Name (username)
                            VALUES(?)''', (name,))
        connection.commit()
        print("Hello", name, ", I will make sure to remember you next time.")
        return
    else:
        print("Welcome back", name)
    connection.close()


if __name__ == "__main__":
    # Configuration and global variables
    CSV_PATH = "COMP3074-CW1-Dataset.csv"
    corpus = read_csv(CSV_PATH)

    # Build the TF-IDF index
    tfidf_matrix, vectorizer = build_index(corpus, "Question")
    print("Yappinator: Hello, how can you help you today \nType help to see what I can do")

    while True:
        user_input = input("> ")
        print("user said", user_input)

        if user_input in ("quit", "stop", "exit"):
            print("Yappinator: Goodbye")
            break

        if user_input == "help":
            continue

        intent = intent_match(user_input)
        print(intent)
        if intent == 'greeting':
            print("Yappinator: Hello, whats your name?")
            name = input("> ")
            identity(name)
        elif intent == 'identity':
            continue
        elif intent == 'question':
            query = preprocess(user_input, remove_stopwords=True)
            similarity_scores = query_similarity(query, tfidf_matrix, vectorizer)
            best_ans = similarity_scores.iloc[0]['Document']
            print(f"Yappinator: ", corpus.loc[best_ans]['Answer'])
            continue
        elif intent == 'small_talk':
            continue
        elif intent == 'discoverability':
            continue
        else:
            print("Yappinator: Sorry I didn't understand that, please try again")

