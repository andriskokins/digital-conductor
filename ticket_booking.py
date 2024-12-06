import datetime

import pandas as pd
from nltk import word_tokenize, pos_tag, ne_chunk
from dateparser.search import search_dates

import classifier
from utils import query_similarity, preprocess, load_model


def book_ticket_stepped():
    while True:
        print("Please enter the city you are travelling from.")
        user_input = input("> ")


def book_ticket(user_input):

    classifier.train_model(split_data=False)

    # Load the city index
    corpus = pd.read_csv('data/gb.csv', usecols=['city'])
    settings = {
        'analyzer': 'char_wb',
        'ngram_range': (3, 5)
    }
    tfidf_matrix, vectorizer = load_model(corpus, 'city', False, settings, "city_names")

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

    # Define confidence thresholds
    high_threshold = 0.7
    medium_threshold = 0.5

    for city in potential_cities:
        query = preprocess(city, remove_stopwords=False)
        similarity_scores = query_similarity(query, tfidf_matrix, vectorizer)
        top_match = corpus.iloc[similarity_scores.index[0]]['city']
        confidence = similarity_scores.iloc[0]['Cosine Similarity']
        print(f"Confidence for '{city}' matching '{top_match}': {confidence}")

        # 3 Tiered confidence confirmation
        if confidence >= high_threshold:
            # High confidence - accept the city
            actual_cities.append(top_match)
        elif confidence >= medium_threshold:
            # Medium confidence - ask for confirmation
            print(f"Did you mean '{top_match}'?")
            user_response = input("> ")
            response_class = classifier.classify_text(user_response)
            if response_class[0] == 'positive':
                actual_cities.append(top_match)
            else:
                print("Please provide the city name again.")
        else:
            # Low confidence - unable to identify
            # print(f"Could not confidently identify the city for '{city}'.")
            continue

    actual_cities = list(set(actual_cities) & set(valid_cities))
    print(f"Actual cities: {actual_cities}")

    ticket_from = actual_cities[0]
    ticket_to = actual_cities[1]
    ticket_date = search_dates(user_input, languages=['en'], settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': datetime.datetime.now()})


    while(ticket_date == None):
        print(f"Please enter a valid date and time. For example, {(datetime.datetime.today()+datetime.timedelta(days=1)).strftime('%d/%m/%Y')} at 10:00")
        user_input = input("> ")
        ticket_date = search_dates(user_input, languages=['en'], settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': datetime.datetime.now()})

    ticket_date = ticket_date[0][1].strftime('%d-%m-%Y %H:%M')
    print(f"I got the following information: From {ticket_from} to {ticket_to} on {ticket_date}, is that correct?")
    user_response = input("> ")
    response_class = classifier.classify_text(user_response)
    if response_class[0] == 'positive':
        return
    else:
        book_ticket_stepped()