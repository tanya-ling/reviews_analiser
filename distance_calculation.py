from prepare_data import read_csv_data
import numpy


def splitting(row):
    splitstring = row['keyword_vector'].split(' ')
    splitstring = [float(s) for s in splitstring]
    return splitstring


def book2book_distance(book_title1, book_title2):
    return book2_distance(book_title1, book_title2) + book2_distance(book_title2, book_title1)


def book2_distance(book_title1, book_title2):
    array1 = numpy.array(db[book_title1])
    array2 = numpy.array(db[book_title2])
    D = 0
    for v in array1:
        mind = 1000
        for w in array2:
            d = numpy.linalg.norm(v-w)
            if d < mind:
                mind = d
        D += mind
    return D


def book2allbooks_distances(book_title):
    dict = {}
    for a in db.index:
        if a != book_title:
            D = book2book_distance(a, book_title)
            dict[a] = D
    return dict


def distance_matrix_creation(sourse_path, results_path):
    db = read_csv_data(sourse_path, '|', rew=False)
    db['keyword_vector'] = db.apply(splitting, axis=1)
    print(db.head())
    db = db.groupby('book_title')['keyword_vector'].agg(list)

if __name__ == "__main__":
    db = read_csv_data('2017_books_v5_preproc_v4_agreg_v1_keywords_v5.csv', '|', rew=False)
    db['keyword_vector'] = db.apply(splitting, axis=1)
    print(db.head())
    db = db.groupby('book_title')['keyword_vector'].agg(list)
    print(db.head())
    print(book2book_distance('A Court of Wings and Ruin', 'American War'))
    print(book2book_distance('American War', 'American War'))
    print(book2book_distance('Wildfire (Hidden Legacy, #3)', 'White Hot (Hidden Legacy, #2)'))
    dict = book2allbooks_distances('Wildfire (Hidden Legacy, #3)')
    print(sorted( ((v,k) for k,v in dict.items()), reverse=False))
