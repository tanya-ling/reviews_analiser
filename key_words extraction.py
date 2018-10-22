import pandas
from prepare_data import read_csv_data, add_column
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


# not good to recompute every time i want to change a number of key-words!
def key_word_extraction(db, kw_numbr):
    docs = db['review'].tolist()
    cv = CountVectorizer(max_df=0.85)
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names = cv.get_feature_names()
    all_keywords = [0 for i in range(len(docs))]
    i = 0
    for doc in docs:
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, kw_numbr)
        all_keywords[i] = ' '.join(keywords)
        i += 1
    return all_keywords


if __name__ == "__main__":
    db = read_csv_data('2017_books_v5_preproc_v3_agreg_v1.csv', '|')
    print(db.info())
    keywords = key_word_extraction(db, 10)
    db = add_column(db, 'keywords', keywords)
    db[['book_title', 'keywords']].to_csv('2017_books_v5_preproc_v3_agreg_v1_keywords_v1.csv', sep='|', index=False)