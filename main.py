import sqlite3

import ticket_booking
import classifier
from utils import preprocess, build_index, query_similarity, load_model

import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk


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
        print(f"Hello {name}, I will make sure to remember you next time.")
        return
    else:
        print("Welcome back", name, '\b! How can I help you today?')
    connection.close()


def create_intent_index():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('intents.csv', header=0,  names=['Intent', 'Pattern'])

    # DO NOT REMOVE STOPWORDS
    # "what can you do" outputs an empty string otherwise
    tfidf_matrix, vectorizer = build_index(df, 'Pattern', False)
    return df, tfidf_matrix, vectorizer


def get_intent(user_input, intent_corpus, intent_matrix, intent_vectorizer):
    user_input = preprocess(user_input, remove_stopwords=False)
    user_input = "".join(user_input)
    print(user_input)

    # Get similarity scores
    similarity_scores = query_similarity(user_input, intent_matrix, intent_vectorizer)

    # Add the intents to the similarity_scores DataFrame
    similarity_scores['Intent'] = intent_corpus['Intent'].iloc[similarity_scores.index].values

    # Print the list of probabilities with intent types
    print(similarity_scores[['Intent', 'Cosine Similarity']][:5])

    # Get the index of the highest scoring pattern
    best_match_idx = similarity_scores.index[0]

    # Get the corresponding intent
    matched_intent = intent_corpus['Intent'].iloc[best_match_idx]

    # Only return the intent if the similarity score is above a threshold
    if similarity_scores.iloc[0]['Cosine Similarity'] > 0.1:
        return matched_intent
    return 'unknown'


def extract_name(text):
    # Check for invalid input
    if text.isdigit() or text.strip() == '':
        print("Yappinator: Please enter a name that is not a digit or empty.")
        input_name = input("> ")
        return extract_name(input_name)

    # Tokenize and tag the text
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    entities = ne_chunk(tagged)
    # print(entities)

    # Look for PERSON entities
    for chunk in entities:
        if hasattr(chunk, 'label') and (chunk.label() == 'PERSON'):
            return ' '.join(c[0] for c in chunk.leaves())

    # Check tags for NNP because it doesn't seem to recognise uncommon names
    for token, tag in tagged:
        if tag == 'NNP' or tag =='NN':
            return token

    # No name found, try again
    return extract_name("")


if __name__ == "__main__":
    classifier.train_model(split_data=False)
    CSV_PATH = "COMP3074-CW1-Dataset.csv"
    CSV_INTENTS = "intents.csv"
    corpus = pd.read_csv(CSV_PATH, index_col='QuestionID')
    intent_corpus = pd.read_csv(CSV_INTENTS, names=['Intent', 'Pattern'])

    # Default Settings
    settings = {}
    # Load or build the search model
    # stops word are removed for the QnA
    tfidf_matrix, vectorizer = load_model(
        corpus=corpus,
        column_name="Question",
        remove_stopwords=True,
        settings=settings,
        model_name="search")
    # Load or build the intent model
    intent_matrix, intent_vectorizer = load_model(
        corpus=intent_corpus,
        column_name="Pattern",
        remove_stopwords=False,
        settings=settings,
        model_name="intent"
    )

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

        elif intent == 'booking':
        #     book a train ticket
            ticket_booking.book_ticket(user_input)

        elif intent == 'identity':
            if username:
                print("Yappinator: Your name is", username)
            else:
                print("Yappinator: I don't know your name yet. Would you like to introduce yourself?")
                save_name(extract_name(input("> ")))

        elif intent == 'question':
            query = preprocess(user_input, remove_stopwords=True)
            similarity_scores = query_similarity(query, tfidf_matrix, vectorizer)
            best_ans = similarity_scores.index[0]
            print(f"Yappinator: {corpus.iloc[best_ans]['Answer']}")

        elif intent == 'small_talk':
            if username:
                print("Yappinator: Im doing well, thanks for asking", username)
            else:
                print("Yappinator: I'm doing well, thanks for asking! How can I help you today?")

        elif intent == 'discoverability':
            print('Yappinator: I can answer some questions, try asking me "What does gringo mean?"')

        else:
            print("Yappinator: Sorry, I didn't understand that. Please try again.")
