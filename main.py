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
    filtered_tokens = [word for word in tokens if word.isalnum()]

    postags = nltk.pos_tag(filtered_tokens, tagset='universal')
    lemmatized_tokens = []

    for word, tag in postags:
        if tag in posmap:
            lemmatized_word = lemmatizer.lemmatize(word, posmap[tag])
        else:
            lemmatized_word = lemmatizer.lemmatize(word)

        if lemmatized_word.isalpha():
            lemmatized_tokens.append(lemmatized_word)

    return lemmatized_tokens


def build_index(data, fields_to_read, is_intent=False):
    """Builds the inverted index across Q&A corpus or intent patterns."""
    index = defaultdict(list)
    doc_count = len(data)

    for doc_id, content in data.items():
        if is_intent:
            # For intent patterns, process each pattern in the list
            terms = []
            if isinstance(content, (list, tuple)) and len(content) > 0:
                # If content is a single list/tuple of patterns
                patterns = content[0] if isinstance(content[0], (list, tuple)) else content
                for pattern in patterns:
                    terms.extend(word_tokenize(pattern.lower()))

            # Calculate term frequencies for this document
            term_frequencies = defaultdict(int)
            for term in terms:
                if term.isalnum():  # Only include alphanumeric terms
                    term_frequencies[term] += 1

            # Calculate weights and add to index
            for term, freq in term_frequencies.items():
                # Calculate IDF: log(total_docs / (1 + num_docs_with_term))
                docs_with_term = len(index[term])
                idf = math.log(doc_count / (1 + docs_with_term)) + 1
                weight = freq * idf
                index[term].append((doc_id, weight))
        else:
            # Original Q&A corpus processing
            for field in fields_to_read:
                text = content.get(field)
                if text:
                    terms = word_tokenize(text.lower())
                    term_frequencies = defaultdict(int)
                    for term in terms:
                        if term.isalnum():
                            term_frequencies[term] += 1
                    for term, freq in term_frequencies.items():
                        idf = math.log(doc_count / (1 + len(index[term]))) + 1
                        weight = freq * idf
                        index[term].append((doc_id, weight))

    return index


# Fixed intent patterns structure
intent_patterns = {
    'identity': ['what is my name','who am i'],
    'greeting': ['hello','hi','hey','good morning','good evening'],
    'question': ['what is', "what are", 'how do', 'can you explain','tell me about','why', 'when','where'],
    'small_talk': ['how are you',"what's up","how's it going", 'how are things'],
    'discoverability': ['what can you do','help','what are your capabilities','what do you know']
}



def load_index(path=SAVE_INDEX_PATH, corpus=None):
    try:
         index = joblib.load(path)
         print(f"Loaded index from {path}")
         return index
    except FileNotFoundError:
         print(f"Index file not found at {path}. Building a new index.")
         return build_index(corpus, ["question", ""])


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


# Example usage:
if __name__ == "__main__":
    fields_to_index = ['greeting', 'small_talk', 'discoverability', 'identity']
    intent_index = build_index(intent_patterns, fields_to_index, is_intent=True)
    save_index(intent_index, "intent.joblib")

    corpus = load_csv(CSV_PATH)
    index = load_index(SAVE_INDEX_PATH, corpus)
    save_index(index, SAVE_INDEX_PATH)

    query = "What are stocks and bonds"
    results = search(query, index, corpus, top_n=3)

    print("Question - ", query+"\n")
    for result in results:
        print(f"answer - {result['answer']}")
        print(f"score - {result['score']}")
        print(f"doc id - {result['doc_id']}")
        print('-'* 30)

    results = search("what are stocks and bonds", intent_index, intent_patterns)

    for result in results:
        print(result)
