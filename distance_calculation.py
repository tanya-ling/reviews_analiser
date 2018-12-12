from prepare_data import read_csv_data
import numpy
import pandas
import time

def splitting(row):
    splitstring = row['keyword_vector'].split(' ')
    splitstring = [float(s) for s in splitstring]
    return splitstring


def change_title(title):
    title = title.replace('вЂ™', 'a€™')
    title = title.replace('????????1? ???????', 'e?Za?«a›?e•·?®?a?—a€•c¬¬1e?? e?•a‚?a‚‹a‚¤a?‡a‚?c·?')
    return title

def book2book_distance(db, book_title1, book_title2):
    return book2_distance(db, book_title1, book_title2) + book2_distance(db, book_title2, book_title1)


def book2_distance(db, book_title1, book_title2):
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


def book2allbooks_distances(db, book_title, number):
    dic = {}
    for a in db.index:
        if a != book_title:
            D = book2book_distance(db, a, book_title)
            dic[a] = D
    list_pairs = [0 for i in range(number)]
    i = 0
    for key, value in sorted(dic.items(), key = lambda x : x[1]):
        list_pairs[i] = [key, value]
        i += 1
        if i >= number:
            return list_pairs


def closest_for_every_book(db, results_path, number=10):
    ids_path = 'books_ids.csv'
    idb = read_csv_data(ids_path, ';')
    idb.set_index('uniqueID', inplace=True)
    dic = {}
    nr = 0
    t0 = time.clock()
    for book_pr in db.index:
        if nr % 72 == 0:
            print(round(nr / 715 * 100), '% done, time consumed', round(-(t0 - time.clock()) / 60), ' min')
        nr += 1
        dic[book_pr] = book2allbooks_distances(db, book_pr, number)
        # dic[idb.loc[change_title(book_pr)]['uniqueID']] = book2allbooks_distances(db, book_pr, number)
    ndf = pandas.DataFrame.from_dict(dic, orient='index')
    ndf.index.name = 'uniqueID'
    ndf.to_csv(results_path, sep='|', index=True)


if __name__ == "__main__":
    # print(book2book_distance('A Court of Wings and Ruin', 'American War'))
    # print(book2book_distance('American War', 'American War'))
    # print(book2book_distance('Wildfire (Hidden Legacy, #3)', 'White Hot (Hidden Legacy, #2)'))
    # dic = book2allbooks_distances('Wildfire (Hidden Legacy, #3)', 3)
    # print(sorted( ((v,k) for k,v in dic.items()), reverse=False))
    sourse_path = 'all_data_less_kw_20.csv'
    db = read_csv_data(sourse_path, '|', rew=False)
    db['keyword_vector'] = db.apply(splitting, axis=1)
    print(db.head())
    db = db.groupby('book_title')['keyword_vector'].agg(list)
    closest_for_every_book('all_data_less_dis_20.csv')
