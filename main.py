import time

from nltk.corpus import stopwords

import ticket_booking
import classifier
import utils
from utils import preprocess, build_index, query_similarity, load_model

import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk


def create_intent_index():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('data/intents.csv', header=0, names=['Intent', 'Pattern'])

    # DO NOT REMOVE STOPWORDS
    # "what can you do" outputs an empty string otherwise
    tfidf_matrix, vectorizer = build_index(df, 'Pattern', False)
    return df, tfidf_matrix, vectorizer


def get_intent(user_input, intent_corpus, intent_matrix, intent_vectorizer):
    user_input = preprocess(user_input, remove_stopwords=False)
    user_input = "".join(user_input)
    # print(user_input)

    # Get similarity scores
    similarity_scores = query_similarity(user_input, intent_matrix, intent_vectorizer)

    # Add the intents to the similarity_scores DataFrame
    similarity_scores['Intent'] = intent_corpus['Intent'].iloc[similarity_scores.index].values

    # Print the list of probabilities with intent types
    # print(similarity_scores[['Intent', 'Cosine Similarity']][:5])

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
        print("Digital Conductor: Please enter a name that is not a digit or empty.")
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
        if tag == 'NNP' or tag == 'NN':
            return token

    # No name found, try again
    return extract_name("")


if __name__ == "__main__":
    classifier.train_model(split_data=False)
    utils.initialize_database()
    CSV_PATH = "data/question_answer.csv"
    CSV_INTENTS = "data/intents.csv"
    corpus = pd.read_csv(CSV_PATH, names=['Question', 'Answer'])
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

    print("Digital Conductor: Hello, how can I help you today?")

    username = ""
    while True:
        user_input = input("> ")

        if user_input.lower() in ("quit", "stop", "exit"):
            print("Digital Conductor: Goodbye")
            break

        start = time.time()
        intent = get_intent(user_input, intent_corpus, intent_matrix, intent_vectorizer)

        if intent == 'greeting':
            if username:
                print(f"Hello {username}")
            else:
                print("Digital Conductor: Hello, what's your name?")
                name_input = input("> ")
                extracted_name = extract_name(name_input)
                if extracted_name:
                    utils.save_name(extracted_name)
                    username = extracted_name
                else:
                    # If NER fails, use the entire input as the name
                    utils.save_name(name_input)
                    username = name_input

        elif intent == 'booking':
            #     book a train ticket
            ticket = ticket_booking.book_ticket(user_input)
            if username:
                print(f"Digital Conductor: {username} your ticket has been booked.")
            utils.save_ticket(username, ticket['departure'], ticket['destination'], ticket['time'])
            end = time.time()
            print(f"Time taken: {end - start:.2f} seconds")

        elif intent == 'ticket':
            if username:
                print(utils.get_tickets(username))
            else:
                print("Digital Conductor: Looks like you haven't introduced yourself yet. Would you like give me the name "
                      "you used to reserve your ticket?")
                print("Digital Conductor: Alternatively, provide your ticket reference number")
                print("Digital Conductor: Reply with a yes for name and no for reference")
                reply = input("> ")
                reply = classifier.classify_text(reply)
                if reply[0] == 'positive':
                    print("Digital Conductor: Great! What's your name please")
                    user_input = input("> ")
                    extracted_name = extract_name(user_input)
                    username = extracted_name
                    print(utils.get_tickets(extracted_name))
                else:
                    print("Digital Conductor: Alright, type in your reference number below")
                    reference_number = int(input("> "))
                    print(utils.get_ticket(reference_number))

        elif intent == 'identity':
            if username:
                print("Digital Conductor: Your name is", username)
            else:
                print("Digital Conductor: I don't know your name yet. Would you like to introduce yourself?")
                utils.save_name(extract_name(input("> ")))

        elif intent == 'question':
            query = preprocess(user_input, remove_stopwords=True)
            similarity_scores = query_similarity(query, tfidf_matrix, vectorizer)
            best_ans = similarity_scores.index[0]
            if (corpus.iloc[best_ans]['Answer'] == 'Answer'):
                print("Digital Conductor: Sorry, I don't have the knowledge to answer that, please try again...")
            else:
                print(f"Digital Conductor: {corpus.iloc[best_ans]['Answer']}")

        elif intent == 'small_talk':
            if username:
                print("Digital Conductor: Im doing well, thanks for asking", username)
            else:
                print("Digital Conductor: I'm doing well, thanks for asking! How can I help you today?")

        elif intent == 'discoverability':
            if username:
                print(f"Digital Conductor: I can help you book a train ticket or search the database for bookings under {username}.")
            else:
                print("Digital Conductor: I can help you book a train ticket, view your ticket, or answer general questions.")

        else:
            print("Digital Conductor: Sorry, I didn't understand that. Please try again.")
