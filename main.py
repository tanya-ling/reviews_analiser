import preprocessing
import key_words_extraction
import distance_calculation
import pandas
from prepare_data import read_csv_data
from ast import literal_eval


if __name__ == "__main__":
    sourth_path = "reviews_final.csv"
    genre= 5
    author = 5
    series = 2
    title = 2
    ill_path = 'ill.csv'
    preproc_path = sourth_path[:-4] + '_agr' + str(genre) + str(author) + str(series) + str(title) + '.csv'

    kw_numbr = 30
    kw_path = preproc_path[:-4] + '_kw' + str(kw_numbr) + '.csv'
    if False:
        db = preprocessing.process_new_file(sourth_path, ill_path, ';', genre, author, series, title)
        # change_weights_only('2017_books_v5_preproc_v3.csv', '2017_books_v5_preproc_v4.csv', '|',
         #                   genre_weight=3, author_weight=4)
        # process_new_lines('2017_books_v5_preproc_v2.csv', '2017_books_v5_preproc_v3.csv', '|')
        print('starting aggregation')
        preprocessing.reviews_aggregation(db, ill_path, preproc_path, '|')
    if False:
        db = read_csv_data(preproc_path, '|')
        model = key_words_extraction.w2v_model_loading()
        columns = ['uniqueID', 'keyword', 'keyword_vector']
        ndf = pandas.DataFrame.from_dict(key_words_extraction.key_word_extraction(db, model, kw_numbr), orient='index')
        ndf.columns = columns
        ndf.to_csv(kw_path, sep='|', index=False)

    bnr = 15
    dis_path = kw_path[:-4] + '_dis' + str(bnr) + '.csv'
    if False:
        db = read_csv_data(kw_path, '|', rew=False)
        db['keyword_vector'] = db.apply(distance_calculation.splitting, axis=1)
        db = db.groupby('uniqueID')['keyword_vector'].agg(list)
        distance_calculation.closest_for_every_book(db, dis_path, bnr)

    # ids_path = 'books_ids.csv'
    # idb = read_csv_data(ids_path, ';')
    # idb.set_index('book_title', inplace=True)

    db = read_csv_data(dis_path, '|', False)
    db.set_index('uniqueID', inplace=True)
    # ndf = pandas.DataFrame(columns=['book_id', 'id_closest', 'place', 'distance'])
    darr = []
    fin_path = 'distances_v20.csv'
    for title, row in db.iterrows():
        t1 = title
        i = 0
        for place in row:
                arr = literal_eval(place)
#            for title2 in arr:
                t2 = arr[0]
                d = arr[1]
                i += 1
                p = i
                d = {'bookID': t1, 'bookID_closest': t2, 'place': p, 'distance': d}
                darr.append(d)
    ndf = pandas.DataFrame.from_dict(darr, orient='columns')
    ndf.to_csv(fin_path, sep=';', index=False, encoding='utf-8')

