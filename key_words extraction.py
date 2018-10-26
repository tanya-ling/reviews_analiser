from prepare_data import read_csv_data, add_column
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time
import pandas
import gensim
# import pyenchant


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn):
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
def key_word_extraction(db, kw_numbr=10):
    docs = db['review'].tolist()
    cv = CountVectorizer(max_df=0.85, stop_words='english')
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names = cv.get_feature_names()
    i = 0
    kw_dict = {}
    j = 0
    for doc in docs:
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, kw_numbr)
        for keyword in keywords:
            wv = word_vector(keyword, model)
            if True:
                kw_dict[j] = [db.at[i, 'book_title'], keyword, ' '.join(map(str, wv))]
                j += 1
        i += 1
    return kw_dict


def w2v_model_loading(path='./GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin'):
    print('starting loading w2vec model')
    t1 = time.clock()
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    print('the model is loaded successfully, time consumed: ', str(time.clock() - t1), ' sec')
    return model


def word_vector(word, model):
    # does not include some stopwords and numbers
    if word in model.vocab:
        return model.wv[word] # .shape(300,))
    return ['1', '2', '3', '4', '5']

if __name__ == "__main__":
    db = read_csv_data('2017_books_v5_preproc_v3_agreg_v1.csv', '|')
    print(db.info())
    model = w2v_model_loading()
    columns = ['book_title', 'keyword', 'keyword_vector']
    ndf = pandas.DataFrame.from_dict(key_word_extraction(db, 10), orient='index')
    ndf.columns = columns
    ndf.to_csv('2017_books_v5_preproc_v3_agreg_v1_keywords_v2.csv', sep='|', index=False)
    print(ndf.info())
