import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split

WORDS_FILE = '../data/word_data.pkl'
AUTHORS_FILE = '../data/email_authors.pkl'


def preprocess(words_file=WORDS_FILE, authors_file=AUTHORS_FILE):
    authors_file_handler = open(authors_file, "rb")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, "rb")
    word_data = pickle.load(words_file_handler)
    words_file_handler.close()

    features_train, features_test, labels_train, labels_test = \
        train_test_split(
            word_data, authors, test_size=0.1, random_state=42
        )

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(
        features_train_transformed).toarray()
    features_test_transformed = selector.transform(
        features_test_transformed).toarray()

    chris_training_email_count = sum(labels_train)
    sara_training_email_count = len(labels_train) - sum(labels_train)
    print('no. of Chris training emails:', chris_training_email_count)
    print('no. of Sara training emails:', sara_training_email_count)

    return (features_train_transformed, features_test_transformed,
            labels_train, labels_test)
