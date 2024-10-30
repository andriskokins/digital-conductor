import os
import joblib
import nltk
import math
from nltk.corpus import stopwords
from nltk.inference.nonmonotonic import print_proof
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pandas as pd


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuration and global variables
DATA_DIR = "data/positive"  # Path to your data directory
SAVE_INDEX_PATH = "index.joblib"  # Path to save the index
CSV_PATH = "COMP3074-CW1-Dataset.csv"

intent_patterns = {
        'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
        'identity': ['what is my name', 'who am i',],
        'question': ['what is', 'how do', 'can you explain', 'tell me about', 'why', 'when', 'where'],
        'small_talk': ['how are you', "what's up", "how's it going", 'how are things'],
        'discoverability': ['what can you do', 'help', 'what are your capabilities', 'what do you know']
    }

# POS tag mapping to WordNet format
posmap = {
    'ADJ': 'a',
    'ADV': 'r',
    'NOUN': 'n',
    'VERB': 'v'
}

# 1. Preprocessing functions
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and
                       word not in stopwords.words('english')]

    postags = nltk.pos_tag(filtered_tokens, tagset='universal')
    lemmatized_tokens = []

    for word, tag in postags:
        if tag in posmap:
            lemmatized_word = lemmatizer.lemmatize(word, posmap[tag])
        else:
            lemmatized_word = lemmatizer.lemmatize(word)

        if lemmatized_word.isalpha() and lemmatized_word not in stopwords.words('english'):
            lemmatized_tokens.append(lemmatized_word)

    return lemmatized_tokens


def build_index(data, fields_to_read):
    """Builds the inverted index across Q&A corpus or intent patterns."""
    index = defaultdict(list)
    doc_count = len(data)

    for doc_id, content in data.items():
        text = content.get('question')
        if text:
            terms = preprocess_text(text)
            term_frequencies = defaultdict(int)
            for term in terms:
                term_frequencies[term] += 1
            for term, freq in term_frequencies.items():
                idf = math.log(doc_count / (1 + len(index[term]))) + 1
                weight = freq * idf
                index[term].append((doc_id, weight))

    return index


def load_index(path=SAVE_INDEX_PATH, corpus=None):
    try:
         index = joblib.load(path)
         print(f"Loaded index from {path}")
         return index
    except FileNotFoundError:
         print(f"Index file not found at {path}. Building a new index.")
         return build_index(corpus, 'question')


def save_index(index, path=SAVE_INDEX_PATH):
    joblib.dump(index, path)
    print(f"Saved index to {path}")


# Calculate cosine similarity
def cosine_similarity(query_vector, document_vector):
    dot_product = sum(query_vector[term] * document_vector.get(term, 0) for term in query_vector)
    query_magnitude = math.sqrt(sum(weight**2 for weight in query_vector.values()))
    doc_magnitude = math.sqrt(sum(weight**2 for weight in document_vector.values()))

    if query_magnitude == 0 or doc_magnitude == 0:
        return 0.0
    return dot_product / (query_magnitude * doc_magnitude)


def load_csv(filepath):
    data = pd.read_csv(filepath)
    corpus = defaultdict(lambda: {"question":None, "answer":None, "document":None})

    for _, row in data.iterrows():
        question_id = row['QuestionID']  # Get the QuestionID
        corpus[question_id]['question'] = row['Question']
        corpus[question_id]['answer'] = row['Answer']
        corpus[question_id]['document'] = row['Document']

    return corpus


def search(query, index, corpus, top_n=5):
    """Searches the index and returns the top N most relevant answers or intent phrases."""
    processed_query = word_tokenize(query.lower())
    query_vector = defaultdict(int)

    for term in processed_query:
        query_vector[term] += 1

    document_scores = defaultdict(float)
    for term, query_weight in query_vector.items():
        if term in index:
            for doc_id, doc_weight in index[term]:
                document_scores[doc_id] += query_weight * doc_weight

    ranked_documents = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)
    top_answers = []

    for doc_id, score in ranked_documents[:top_n]:
        if isinstance(corpus[doc_id], dict):  # Q&A corpus
            answer = corpus[doc_id].get('answer', "No answer found.")
            top_answers.append({"answer": answer, "score": score, "doc_id": doc_id})
        elif isinstance(corpus[doc_id], list):  # Intent patterns
            # Join intent phrases as a single response
            intent_response = " / ".join(corpus[doc_id])
            top_answers.append({"answer": intent_response, "score": score, "doc_id": doc_id})

    return top_answers


def intent_match(input):
    input_text = input.lower().strip()


if __name__ == "__main__":
    # Load Corpus from CSV
    corpus = load_csv(CSV_PATH)
    # Build or load index
    index = load_index(SAVE_INDEX_PATH, corpus)
    # save_index(index, SAVE_INDEX_PATH)

    # fields_to_index = ['question', 'answer', 'greeting', 'small_talk', 'discoverability', 'identity']
    # intent_index = build_index(intent_patterns, fields_to_index)
    # save_index(intent_index, "intent.joblib")

    query = "how did anne frank die"
    top_answers = search(query, index, corpus, top_n=3)  # Get top 3 answers

    # for result in top_answers:
    #     print(result)

    for result in top_answers:
        print(f"Document ID: {result['doc_id']}")
        print(f"Score: {result['score']}")
        print(f"Answer: {result['answer']}")
        print("-" * 20)

    print("0")
