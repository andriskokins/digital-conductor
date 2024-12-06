from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from utils import preprocess

# Module-level variables to store the trained components
classifier = None
count_vect = None
tfidf_transformer = None


def evaluate_model(X_test, y_test):
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = classifier.predict(X_new_tfidf)
    print(confusion_matrix(y_test, predicted))
    print(accuracy_score(y_test, predicted))
    print(f1_score(y_test, predicted, pos_label='positive'))

def load_data(label_dir):

    data = []
    labels = []
    for label, filepath in label_dir.items():
        with open(filepath, encoding='utf8', errors='ignore') as f:
            for line in f:
                processed = preprocess(line, remove_stopwords=False)
                data.append(processed)
                labels.append(label)
    return data, labels

def train_model(split_data=False):
    global classifier, count_vect, tfidf_transformer

    label_dir = {
        "positive": "data/positive.csv",
        "negative": "data/negative.csv",
    }

    data, labels = load_data(label_dir)
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels,
                                                            test_size=0.25, random_state=42)
    else:
        X_train, X_test, y_train, y_test = data, [], labels, []

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
    X_train_tf = tfidf_transformer.fit_transform(X_train_counts)

    classifier = LogisticRegression(random_state=0).fit(X_train_tf, y_train)

    # Optional evaluation
    if split_data:
        evaluate_model(X_test, y_test)


def classify_text(text):
    preprocessed_input = preprocess(text, remove_stopwords=False)
    processed_newdata = count_vect.transform([preprocessed_input])
    processed_newdata = tfidf_transformer.transform(processed_newdata)
    prediction = classifier.predict(processed_newdata)
    probabilities = classifier.predict_proba(processed_newdata)
    labeled_probabilities = {
        label: prob for label, prob in zip(classifier.classes_, probabilities[0])
    }
    return prediction[0], labeled_probabilities
