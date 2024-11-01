import nltk
import pandas as pd
from nltk import WordNetLemmatizer, word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np

nltk.download('maxent_ne_chunker_tab')
nltk.download('words')


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


def build_index(data, column, stopwords):
    questions = data.loc[:, column]

    # Apply preprocessing to each question in the corpus
    # do not remove stops words for intent matrix
    questions = questions.apply(lambda q: preprocess(q, remove_stopwords=stopwords))

    # Create a TF-IDF term-document matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)

    return tfidf_matrix, vectorizer


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


def save_name(name):
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
        print("Howdy", name, '!')
    connection.close()


def create_intent_index():
    # can add more patterns for better intent matching
    data = {
        'Pattern': [
            'hello', 'hi', 'hey', 'good morning', 'good evening',
            'what is my name', 'who am i',
            'how are you', "what's up", "how's it going", 'how are things',
            'what can you do', 'help', 'what are your capabilities', 'what do you know',
            'what is', 'what are', 'how do', 'how'
        ],
        'Intent': [
            'greeting', 'greeting', 'greeting', 'greeting', 'greeting',
            'identity', 'identity',
            'small_talk', 'small_talk', 'small_talk', 'small_talk',
            'discoverability', 'discoverability', 'discoverability', 'discoverability',
            'question', 'question', 'question', 'question'
        ]
    }
    df = pd.DataFrame(data)
    # DO NOT REMOVE STOPWORDS
    # "what can you do" outputs an empty string otherwise
    tfidf_matrix, vectorizer = build_index(df, 'Pattern', False)
    return df, tfidf_matrix, vectorizer


def get_intent(user_input, intent_corpus, intent_matrix, intent_vectorizer):
    # Get similarity scores
    similarity_scores = query_similarity(user_input, intent_matrix, intent_vectorizer)

    # Get the index of the highest scoring pattern
    best_match_idx = similarity_scores.index[0]

    # Get the corresponding intent
    matched_intent = intent_corpus.iloc[best_match_idx]['Intent']

    # Only return the intent if the similarity score is above a threshold
    if similarity_scores.iloc[0]['Cosine Similarity'] > 0.1:
        return matched_intent
    return 'unknown'


def extract_name(text):
    # Tokenize and tag the text
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Use NLTK's named entity chunker
    entities = ne_chunk(tagged)

    # Look for PERSON entities
    for chunk in entities:
        if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
            return ' '.join(c[0] for c in chunk.leaves())


if __name__ == "__main__":
    CSV_PATH = "COMP3074-CW1-Dataset.csv"
    corpus = read_csv(CSV_PATH)

    # stops word are removed for the QnA
    tfidf_matrix, vectorizer = build_index(corpus, "Question", True)
    intent_corpus, intent_matrix, intent_vectorizer = create_intent_index()

    print("Yappinator: Hello, how can I help you today?")

    username = ""
    while True:
        user_input = input("> ")

        if user_input.lower() in ("quit", "stop", "exit"):
            print("Yappinator: Goodbye")
            break

        intent = get_intent(user_input, intent_corpus, intent_matrix, intent_vectorizer)

        if intent == 'greeting':
            if username:
                print(f"Hello {username}")
            else:
                print("Yappinator: Hello, what's your name?")
                name_input = input("> ")
                extracted_name = extract_name(name_input)
                if extracted_name:
                    save_name(extracted_name)
                    username = extracted_name
                else:
                    # If NER fails, use the entire input as the name
                    save_name(name_input)
                    username = name_input

        elif intent == 'identity':
            if username:
                print("Yappinator: Your name is", username)
            else:
                print("Yappinator: I don't know your name yet. Would you like to introduce yourself?")
                save_name(input("> "))

        elif intent == 'question':
            query = preprocess(user_input, remove_stopwords=True)
            similarity_scores = query_similarity(query, tfidf_matrix, vectorizer)
            best_ans = similarity_scores.index[0]
            print(f"Yappinator: {corpus.iloc[best_ans]['Answer']}")

        elif intent == 'small_talk':
            print("Yappinator: I'm doing well, thanks for asking! How can I help you today?")

        elif intent == 'discoverability':
            print('Yappinator: I can answer some questions, try asking me "What does gringo mean?"')

        else:
            print("Yappinator: Sorry, I didn't understand that. Please try again.")
