import prepare_data as pd


def process_new_file(path_old, name_new, weights):
    # weighs is an length 4 list with values: genre_weight, author_weight, series_weight, title_weight
    db = pd.read_csv_data(path_old)
    db = db[db.review.notnull()]
    db = pd.add_column(db, 'stemmed_info', pd.stem_comments(db.review))
    db = pd.add_column(db, 'combined_info', pd.combine_information(db, weights[0], weights[1], weights[2], weights[3]))
    print(db.info())
    with open(name_new + '.csv', mode='w', newline='\r\n') as f:
        db.to_csv(f, sep="|", line_terminator='\r\n', encoding='utf-8')


def process_new_lines(path_old, name_new, weights):
    # weighs is an length 4 list with values: genre_weight, author_weight, series_weight, title_weight
    db = pd.read_csv_data(path_old)
    db = db[db.review.notnull()]
    for index, row in db.iterrows():
        if pandas.isnull(row['stemmed_info']):
            row['stemmed_info'] = pd.lemmatize_one_comment(row['review'])
            row['combined_info'] = pd.combine_info_row(row, weights[0], weights[1], weights[2], weights[3])
    print(db.info())
    db.to_csv(name_new + '.csv', sep='|')


def change_weights_only(path_old, name_new, weights):
    # weighs is an length 4 list with values: genre_weight, author_weight, series_weight, title_weight
    db = pd.read_csv_data(path_old)
    db = db[db.review.notnull()]
    for index, row in db.iterrows():
        row['combined_info'] = pd.combine_info_row(row, weights[0], weights[1], weights[2], weights[3])
    print(db.info())
    db.to_csv(name_new + '.csv', sep='|')


if __name__ == "__main__":
    process_new_file("2017_books_v5.csv", '2017_books_v5_preproc_v1', [5, 5, 5, 5])