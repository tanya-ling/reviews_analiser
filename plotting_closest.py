import matplotlib.pylab as plt
from prepare_data import read_csv_data
from random import uniform
from math import pi, sin, cos, log, sqrt
from ast import literal_eval


def bold_title(title):
    title = title.replace('#', '\#')
    return r"$\bf{" + title + "}$"


def creater_coordinates(data, title):
    dic = {bold_title(title): [0, 0]}
    angle_r = 0.3 * pi
    for item in data:
        angle = uniform(0, 2) * pi
        d = log(literal_eval(item)[1])
        d = literal_eval(item)[1]
        d = sqrt(literal_eval(item)[1] )
        dic[literal_eval(item)[0].replace('(', '\n(')] = [d * sin(angle), d * cos(angle)]
        dic[literal_eval(item)[0].replace('(', '\n(')] = [d * sin(angle_r), d * cos(angle_r)]
        angle_r += 0.2 * pi
    return dic


def plotting(test, book_title):
    xs, ys = zip(*test.values())
    labels = test.keys()

    plt.figure(figsize=(8, 10))
    plt.title("The closest books to: " + bold_title(book_title), fontsize=20)
    plt.scatter(xs, ys, marker='o')
    plt.axis('scaled')
    for label, x, y in zip(labels, xs, ys):
        plt.annotate(label, xy=(x, y), ha='center')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('foo.png')


if __name__ == "__main__":
    sourse_path = 'all_data_distances_v1.csv'
    db = read_csv_data(sourse_path, '|', rew=False)
    db.set_index('book_title', inplace=True)
    title = 'Harry Potter and the Sorcerer\'s Stone'
    dic = creater_coordinates(db.loc[title].values, title)
    plotting(dic, title)
