from datetime import datetime, timedelta, time

import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk
from dateparser.search import search_dates

import classifier
import utils
from utils import query_similarity, preprocess, load_model


# Load the city index
corpus = pd.read_csv('data/uk_cities.csv', usecols=['city'])
settings = {
    'analyzer': 'char_wb',
    'ngram_range': (3, 5)
}

classifier.train_model(split_data=False)
tfidf_matrix, vectorizer = load_model(corpus, 'city', False, settings, "city_names")


def book_ticket_stepped():
    actual_cities = []

    print("Im sorry I couldn't get it right the first time\nLet's try one step at a time")
    print("Enter your departure city, note this needs to be a city within the UK, e.g. London")
    user_input = input("> ")
    actual_cities.append(confidence(user_input))
    print("Great, we're half way there! Now enter your destination city.")
    user_input = input("> ")
    actual_cities.append(confidence(user_input))
    print("Now we just need a suitable date and time for your travel")
    user_input = input("> ")

    ticket = {
        'departure': actual_cities[0],
        'destination': actual_cities[1],
        'time': get_time(user_input)[0][1]
    }
    ticket_date = ticket['time'].strftime('%d-%m-%Y %H:%M')

    print(f"I got the following information: From {ticket['departure']} to {ticket['destination']} on {ticket_date}, is that correct?")
    user_response = input("> ")
    response_class = classifier.classify_text(user_response)
    if response_class[0] == 'positive':
        return ticket
    else:
        print("Here we go again")
        book_ticket_stepped()


def get_time(text):
    next_day = datetime.now() + timedelta(days=1)

    while True:
        current_time = datetime.now()
        ticket_date = search_dates(text, languages=['en'], settings={ 'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': datetime.now()})

        if ticket_date:
            parsed_date = ticket_date[0][1]

            # Check if date is at least 30 minutes in the future
            if (parsed_date - current_time) < timedelta(minutes=30):
                print("Please enter a date and time at least 30 minutes from now.")
            # Check if time is between 06:00 and 23:00
            elif time(23, 0) <= parsed_date.time() or parsed_date.time() <= time(6, 0):
                print("Please enter a time between 06:00 and 23:00.")
            else:
                return ticket_date
        else:
            print(f"Please enter a valid date and time. For example, {next_day.strftime('%d/%m/%Y')} at 10:00")

        # Prompt the user for input again
        text = input("> ")


def confidence(text):
    # Define confidence thresholds
    high_threshold = 0.7
    medium_threshold = 0.5

    query = preprocess(text, remove_stopwords=False)
    similarity_scores = query_similarity(query, tfidf_matrix, vectorizer)
    top_match = corpus.iloc[similarity_scores.index[0]]['city']
    conf_value = similarity_scores.iloc[0]['Cosine Similarity']
    print(f"Confidence for '{text}' matching '{top_match}': {conf_value}")

    if conf_value >= high_threshold:
        # High confidence - accept the city
        return top_match
    elif conf_value >= medium_threshold:
        # Medium confidence - ask for confirmation
        print(f"Did you mean '{top_match}'?")
        user_response = input("> ")
        response_class = classifier.classify_text(user_response)
        if response_class[0] == 'positive':
            return top_match
        else:
            print("Sorry, I didn't understand, enter the name again")
            user_response = input("> ")
            return confidence(user_response)
    # else:
    #     print("Sorry, I didn't understand, enter the name again")
    #     user_response = input("> ")
    #     return confidence(user_response)


def book_ticket(user_input, attempts=0):

    valid_cities = corpus['city'].tolist()
    potential_cities = []
    actual_cities = []

    tokens = word_tokenize(user_input)
    tagged = pos_tag(tokens)
    entities = ne_chunk(tagged)

    potential_cities += [word for word, pos in tagged if pos in ('NN', 'NNS', 'NNP', 'NNPS', 'VB')]

    # Look for GPE - only works properly for well-known places and needs to be capitalized
    for chunk in entities:
        if hasattr(chunk, 'label') and (chunk.label() == 'GPE'):
            city = ' '.join(c[0] for c in chunk.leaves())
            if city not in potential_cities:
                potential_cities.append(city)
    print(f"Potential cities: {potential_cities}")

    for city in potential_cities:
        actual_cities.append(confidence(city))

    actual_cities = [city for city in actual_cities if city in valid_cities]
    print(f"Actual cities: {actual_cities}")

    if (attempts <= 1):
        if (len(actual_cities) < 2):
            attempts += 1
            print("Yappinator: To book a ticket, enter the departure city, then the destination city, followed by a suitable date and time.")
            print("Yappinator: For example 'I want to travel from London to Manchester tomorrow at 10:00'")
            user_input = input("> ")
            return book_ticket(user_input, attempts)
        # elif (len(actual_cities) > 2):
        #     attempts += 1
        #     print("Yappinator: I can only book tickets for two cities at a time.")
        #     print("Yappinator: For example 'I want to travel from London to Manchester tomorrow at 10:00'")
        #     user_input = input("> ")
        #     return book_ticket(user_input, attempts)
    elif (len(actual_cities) != 2):
        return book_ticket_stepped()

    ticket = {
        'departure': actual_cities[0],
        'destination': actual_cities[1],
        'time': get_time(user_input)[0][1]
    }

    ticket_date = ticket['time'].strftime('%d-%m-%Y %H:%M')
    print(f"I got the following information: From {ticket['departure']} to {ticket['destination']} on {ticket_date}, is that correct?")
    user_response = input("> ")
    response_class = classifier.classify_text(user_response)
    if response_class[0] == 'positive':
        return ticket
    else:
        book_ticket_stepped()
