import os
import re
import sys
import json
from itertools import izip

from glob import glob
from sklearn.externals import joblib
from sklearn.datasets.base import Bunch

import scraper


def get_data(data_path):
    
    all_data = []

    for path in glob(os.path.join(data_path, '*.json')):
        with open(path, 'r') as jsonfile:
            data = json.loads(jsonfile.read())
            for article in data.get('articles'):
                all_data.extend([scraper.clean(article['content'])])

    return Bunch(categories=scraper.CATEGORIES.keys(),
                 values=None,
                 data=all_data)


def main(path):

    
    files = glob(os.path.join(path, '*.pkl'))
    filename = max(files, key=lambda x: int(re.sub(r'\D', '', x)))

    
    if not filename:
        print "No models found in %s" % path
        sys.exit(1)
    model = joblib.load(filename)
    data = get_data('input')
    data_weighted = model['vectorizer'].transform(data.data)
    data_weighted = model['feature_selection'].transform(data_weighted)
    prediction = model['clf'].predict(data_weighted)
    for text, prediction in izip(data.data, prediction):
        print scraper.CATEGORIES.keys()[prediction].ljust(15, ' '), text[:100], '...'

if __name__ == '__main__':
    main('training')
