from typing import Set, Any, Callable

import time

from copy import deepcopy

import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from tqdm import tqdm


def u_time(func: Callable):

    def call(*args, **kwargs):

        start = time.time()

        ret = func(*args, **kwargs)

        end = time.time()

        print('Execution Time : %.2f seconds' % (end - start))

        return ret

    return call


def moving_difference(values, skip=1):

    if len(values) < 2:

        diff = deepcopy(values)
        diff[:skip] = 0.0

        return diff

    window = np.array([1, -1])

    diff = np.convolve(values, window, 'same')

    diff[:skip] = 0.0

    return diff


# early_stop=True is more efficient for <---> suppress_ugly_pattern(...)
def ugly_pattern(sstr, early_stop=False):

    pattern = ''

    loc = []

    for i, char in enumerate(sstr):

        pattern += char

        loc.append(sstr.count(pattern) - 1)

        if loc[i] > 0:

            sstr = sstr[len(pattern) - 1:]

            pattern = char

            loc[i] = sstr.count(pattern) - 1

            if early_stop:

                return loc

    return loc


def suppress_ugly_pattern(sstr):

    loc = ugly_pattern(sstr, early_stop=False)

    for i in range(len(loc) - 1):

        if loc[i] != loc[i + 1] and sstr[i+1:].startswith(sstr[:i + 1]):

            return suppress_ugly_pattern(sstr[i + 1:])

    return sstr


def bag_of_words(sequences: Any, words: Set[str], return_counts=True):

    counts = {word: [] for word in words}

    for i in tqdm(range(len(sequences))):

        for word in words:

            if return_counts:

                counts[word].append(sequences[i].count(word))

            else:

                counts[word].append(word in sequences[i])

    counts = pd.DataFrame(counts)

    return counts


def find_common_words(sequences: Any, min_length=2, quantile=0.999):

    text = Text()

    text.compile()

    for i in tqdm(range(len(sequences))):

        text.next(sequences[i], min_length=min_length)

    return text.common(quantile=quantile)


def label_encoder(dataframe, columns):

    for key in columns:

        dataframe[key] = LabelEncoder().fit_transform(dataframe[key].values)

    return dataframe


def label_binarizer(dataframe, column):

    unique = np.unique(dataframe[column])

    name = list(map(lambda col_name: column + '_' + str(col_name), unique))

    dataframe[name] = LabelBinarizer().fit_transform(dataframe[column])

    return dataframe


class Text:

    def __init__(self):

        self.re_filters = {}
        self.re_lang = None

        self.dictionary = dict()

    def compile_filters(self):

        with open('datasets/filters.txt', 'r') as buffer:

            filters = buffer.readline().split(' ')

        pattern = r''

        for i, char in enumerate(filters):

            pattern += '\{}+'.format(char)

            if i != len(filters) - 1:

                pattern += '|'

        self.re_filters['symbols'] = re.compile(pattern)
        self.re_filters['space'] = re.compile(' +')
        self.re_filters['urls'] = re.compile(r'(https?://\S+)')

    def compile_lang(self):

        self.re_lang = re.compile(r'\w+')

    def compile(self):

        self.compile_filters()
        self.compile_lang()

    def apply_filters(self, sentence: str, lower: bool = True):

        sentence = self.re_filters['urls'].sub(' ', sentence)
        sentence = self.re_filters['symbols'].sub('', sentence)
        sentence = self.re_filters['space'].sub(' ', sentence)

        if lower:

            sentence = sentence.lower()

        return sentence

    def match_words(self, sentence: str, lower: bool = True, join: bool = False):

        sentence = self.apply_filters(sentence, lower=lower)
        sentence = self.re_lang.findall(sentence)

        if join:

            sentence = ' '.join(sentence)

        return sentence

    def next(self, sentence: str, min_length: int = 1, unique=True):

        words = self.match_words(sentence)

        if unique:

            words = set(words)

        for word in words:

            # suppress repeated substring
            word = suppress_ugly_pattern(word)

            if len(word) < min_length:

                continue

            if word not in self.dictionary:

                self.dictionary[word] = 0

            self.dictionary[word] += 1

    def next_unique(self, sentence, words: Set[str]):

        for word in words:

            if word not in self.dictionary:

                self.dictionary[word] = 0

            self.dictionary[word] += sentence.count(word)

    def common(self, quantile: float = 0.95):

        common_words = set()

        frequency = np.quantile(list(self.dictionary.values()), q=quantile)
        frequency = np.round(frequency, decimals=2)

        for word in self.dictionary:

            if self.dictionary[word] >= frequency:

                common_words.add(word)

        return frequency, common_words

    def reset(self):

        self.dictionary = dict()
