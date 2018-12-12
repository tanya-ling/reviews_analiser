from prepare_data import read_csv_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time
import pandas
import gensim
from autocorrect import spell


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


def kw_vectorization(db, tfidf_transformer, cv, docs, feature_names, kw_numbr, model, j=0, i=0):
    kw_dict = {}
    nr = 0
    t0 = time.clock()
    for doc in docs:
        if nr % 70 == 0:
            print(round(nr / 722 * 100), '% done, time consumed', -round(t0 - time.clock()), ' sec')
        nr += 1
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, kw_numbr)
        number_unknown = 0
        first_kw = True
        for keyword in keywords:
            kw = spell(keyword)
            wv = word_vector(kw, model)
            if kw in model.vocab:
                kw_dict[j] = [db.at[i, 'uniqueID'], kw, ' '.join(map(str, wv))]
                if first_kw:
                    first_kw = False
                    main_kw = kw_dict[j]
                j += 1
            else:
                number_unknown += 1
        for word in range(number_unknown):
            kw_dict[j] = main_kw
            j += 1
        i += 1
    return kw_dict


# not good to recompute every time i want to change a number of key-words!
def key_word_extraction(db, model, kw_numbr=10):
    docs = db['review'].tolist()
    cv = CountVectorizer(max_df=0.85, stop_words='english')
    t = time.clock()
    word_count_vector = cv.fit_transform(docs)
    t1 = time.clock()
    print('documents are fit transformed, time consumed: ', t1 - t, ' sec')
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names = cv.get_feature_names()
    t2 = time.clock()
    kw_dict = kw_vectorization(db, tfidf_transformer, cv, docs, feature_names, kw_numbr, model)
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
        return model.wv[word]  # .shape(300,))
    return ['1', '2', '3', '4', '5']


if __name__ == "__main__":
    db = read_csv_data('all_data_preproc_v3_agreg_v2.csv', '|')
    print(db.info())
    model = w2v_model_loading()
    columns = ['book_title', 'keyword', 'keyword_vector']
    ndf = pandas.DataFrame.from_dict(key_word_extraction(db, 20), orient='index')
    ndf.columns = columns
    ndf.to_csv('all_data_preproc_v1_agreg_v3_keywords_v2_20.csv', sep='|', index=False)
    print(ndf.info())
