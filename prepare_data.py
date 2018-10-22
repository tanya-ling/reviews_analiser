import pandas
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


def read_csv_data(name, deli):
    csv_path = name
    with open(csv_path, "r") as f_obj:
        db = csv_reader(f_obj, deli)
    return db


def csv_reader(file_obj, deli):
    """
    Read a csv file
    """
    reader = pandas.read_csv(file_obj, sep=deli)
    reader = reader[reader.review.notnull()]
    reader['review'] = reader.apply(delete_n, axis=1)
    print(reader.info())
    return reader


def delete_n(row):
    return row['review'].replace('\n', ' ')


def add_column(df, name_col, list_to_add):
    if name_col == 'stemmed_info':
        df = df.assign(stemmed_info=list_to_add)
    elif name_col == 'combined_info':
        df = df.assign(combined_info=list_to_add)
    elif name_col == 'keywords':
        df = df.assign(keywords=list_to_add)
    return df


def lemmatize_one_comment(comment):
    wnl = WordNetLemmatizer()
    return ' '.join([wnl.lemmatize(i.lower(), j[0].lower()) if j[0].lower() in ['a', 'n', 'v']
                  else wnl.lemmatize(i.lower()) for i, j in pos_tag(word_tokenize(comment))])


def stem_comments(column):
    tokenized = [lemmatize_one_comment(comment) for comment in column]
    return tokenized


def combine_info_row(row, genre_weight, author_weight, series_weight, title_weight):
    # if genre_weight == 1:
    #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
    if pandas.isnull(row['Book_series']):
        series_info = ' '
    else:
        series_info = row['Book_series'].replace('0', '')
        for j in list('123456789'):
            series_info = series_info.replace(j, '')
    genre_info = row['genre'].lower().replace(',', '')
    info = ' '.join([(row['book_title'].lower() + ' ') * title_weight, (genre_info + ' ') * genre_weight,
                     (row['book_author'].lower() + ' ') * author_weight, (series_info + ' ') * series_weight,
                     row['stemmed_info']])
    return info


def combine_information(df, genre_weight, author_weight, series_weight, title_weight):
    slength = len(df['book_author'])
    infos = [0 for i in range(slength)]
    i = 0
    for index, row in df.iterrows():
        info = combine_info_row(row, genre_weight, author_weight, series_weight, title_weight)
        infos[i] = info
        i += 1
    return infos


def text_sum(text_series):
    x = ''
    for i in text_series:
        x += i + '0'
    return x

if __name__ == "__main__":
    db = read_csv_data("2017_books_v5.csv", ';')
    db = add_column(db, 'stemmed_info', stem_comments(db.review))
    db = add_column(db, 'combined_info', combine_information(db, 5, 5, 5, 5))
    db.to_csv('2017_books_v5_preproc_v1.csv', sep='|')
    print(db.info())


# titles_set = set(db['book_title'])
# print('unic book_titles:', len(db['book_title'].unique().tolist()))

# print(db.ix[2940]['book_title'])

# review_exist = []
# for i in range(2940):
#     if pandas.isnull(db.at[i,'book_genre']):
#         pass
#     else:
#         review_exist.append(db.at[i, 'book_title'])
#
# print(len(set(review_exist)))
