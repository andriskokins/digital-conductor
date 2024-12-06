import datetime

import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk
from dateparser.search import search_dates

import classifier
from utils import query_similarity, preprocess, load_model

# Load the city index
corpus = pd.read_csv('data/gb.csv', usecols=['city'])
settings = {
    'analyzer': 'char_wb',
    'ngram_range': (3, 5)
}

tfidf_matrix, vectorizer = load_model(corpus, 'city', False, settings, "city_names")


def book_ticket_stepped():
    actual_cities = []

    print("Im sorry I couldn't get it right the first time\nLet's try one step at a time")
    print("Enter your departure city, note this needs to be a major city within the UK, e.g. London")
    user_input = input("> ")
    actual_cities.append(confidence(user_input))
    print("Great, we're half way threre! Now enter your destination city.")
    user_input = input("> ")
    actual_cities.append(confidence(user_input))
    print("Now we just need a suitable date and time for your travel")
    user_input = input("> ")

    ticket_from = actual_cities[0]
    ticket_to = actual_cities[1]
    ticket_date = get_time(user_input)
    ticket_date = ticket_date[0][1].strftime('%d-%m-%Y %H:%M')

    print(f"I got the following information: From {ticket_from} to {ticket_to} on {ticket_date}, is that correct?")
    user_response = input("> ")
    response_class = classifier.classify_text(user_response)
    if response_class[0] == 'positive':
        return
    else:
        print("Here we go again")
        book_ticket_stepped()


def get_time(text):
    ticket_date = search_dates(text, languages=['en'],
                               settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': datetime.datetime.now()})

    current_day = datetime.datetime.now()
    next_day = current_day + datetime.timedelta(days=1)
    while (ticket_date == None):
        print(f"Please enter a valid date and time. For example, {next_day.strftime('%d/%m/%Y')} at 10:00")
        user_input = input("> ")
        ticket_date = search_dates(user_input, languages=['en'],
                                   settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': current_day})
    return ticket_date


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
    else:
        print("Sorry, I didn't understand, enter the name again")
        user_response = input("> ")
        return confidence(user_response)


def book_ticket(user_input):
    classifier.train_model(split_data=False)

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
            potential_cities.append(city)
    print(f"Potential cities: {potential_cities}")

    for city in potential_cities:
        actual_cities.append(confidence(city))

    actual_cities = list(set(actual_cities) & set(valid_cities))
    print(f"Actual cities: {actual_cities}")

    if len(actual_cities) == 0:
        book_ticket_stepped()

    ticket_from = actual_cities[0]
    ticket_to = actual_cities[1]
    ticket_date = get_time(user_input)

    ticket_date = ticket_date[0][1].strftime('%d-%m-%Y %H:%M')
    print(f"I got the following information: From {ticket_from} to {ticket_to} on {ticket_date}, is that correct?")
    user_response = input("> ")
    response_class = classifier.classify_text(user_response)
    if response_class[0] == 'positive':
        return
    else:
        book_ticket_stepped()
