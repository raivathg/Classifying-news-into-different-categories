
Using all the articles scraped by ``scrapper.py`` create a dataset and
train the classifier. This script generate a ``.pkl`` we'll use later to
classify, articles without category.

Usage
-----
$ python train.py



import os
import json
import time
from collections import OrderedDict

from glob import glob
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib


def get_data(articles_path):
    values, all_data, categories = [], OrderedDict(), []

    for path in glob(os.path.join(articles_path, '*.json')):
        category = os.path.basename(path).rsplit('.', 1)[0]
        with open(path, 'r') as jsonfile:

            data = json.loads(jsonfile.read())
            categories.append(category)
            all_data[category] = data

    
    best_category = min([len(c) for c in all_data.values()])

    
    values = [[i] * best_category for i in xrange(len(categories))]

    join = lambda x, y: x + y
    return Bunch(categories=categories,
                 values=reduce(join, values),
                 data=reduce(join, [c[:best_category] for c in all_data.values()]))


def main(path):
    datatrain = get_data(path)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', max_features=6000,
                                 strip_accents='unicode')
    data_weighted = vectorizer.fit_transform(datatrain.data)

    
    feature_selection = SelectPercentile(f_classif, percentile=20)
    data_weighted = feature_selection.fit_transform(data_weighted, datatrain['values'])
    clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
    clf.fit(data_weighted, datatrain['values'])
    if not os.path.exists('training'):
        os.mkdir('training')

    filename = 'training/{0}.pkl'.format(int(time.time()))
    joblib.dump({'clf': clf,
                 'vectorizer': vectorizer,
                 'feature_selection': feature_selection}, filename, compress=9)

if __name__ == '__main__':
    main('articles')
