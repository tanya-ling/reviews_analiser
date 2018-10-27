import prepare_data as pd
import pandas


def process_new_file(path_old, path_new, old_deli, genre_weight=1, author_weight=1, series_weight=1, title_weight=1):
    db = pd.read_csv_data(path_old, old_deli)
    db = pd.add_column(db, 'stemmed_info', pd.stem_comments(db.review))
    db = pd.add_column(db, 'combined_info', pd.combine_information(db, genre_weight, author_weight,
                                                                   series_weight, title_weight))
    print(db.info())
    with open(path_new, mode='w', newline='\r\n') as f:
        db.to_csv(f, sep="|", line_terminator='\r\n', index=False)


def process_new_lines(path_old, path_new, old_deli, genre_weight=1, author_weight=1, series_weight=1, title_weight=1):
    db = pd.read_csv_data(path_old, old_deli)
    for index, row in db.iterrows():
        if pandas.isnull(row['stemmed_info']):
            db.at[index, 'stemmed_info'] = pd.lemmatize_one_comment(row['review'])
            row['stemmed_info'] = db.at[index, 'stemmed_info']
            db.at[index, 'combined_info'] = pd.combine_info_row(row, genre_weight, author_weight,
                                                                   series_weight, title_weight)
    print(db.info())
    db.to_csv(path_new, sep='|', index=False)


def change_weights_only(path_old, path_new, old_deli, genre_weight=1, author_weight=1, series_weight=1, title_weight=1):
    db = pd.read_csv_data(path_old, old_deli)
    for index, row in db.iterrows():
        db.at[index, 'combined_info'] = pd.combine_info_row(row, genre_weight, author_weight,
                                                                   series_weight, title_weight)

    print(db.info())
    db.to_csv(path_new, sep='|', index=False, encoding='cp1251')


def reviews_aggregation(path_old, path_new, old_deli):
    db = pd.read_csv_data(path_old, old_deli)
    db = db[['book_title', 'combined_info', 'ID']]
    db = db.groupby('book_title')['combined_info'].agg(pd.text_sum).reset_index()
    # db = db.groupby('book_title')['combined_info'].agg(pd.text_sum)
    # print(db.info())
    db.columns = ['book_title', 'review']
    db.to_csv(path_new, sep='|', index=False, encoding='cp1251')


if __name__ == "__main__":
    # process_new_file("2017_books_v5.csv", '2017_books_v5_preproc_v1.csv', ';', 5, 5, 5, 5)
    change_weights_only('2017_books_v5_preproc_v3.csv', '2017_books_v5_preproc_v4.csv', '|',
                        genre_weight=3, author_weight=4)
    # process_new_lines('2017_books_v5_preproc_v2.csv', '2017_books_v5_preproc_v3.csv', '|')
    print('weights changed, starting aggregation')
    reviews_aggregation('2017_books_v5_preproc_v4.csv', '2017_books_v5_preproc_v4_agreg_v1.csv', '|')
