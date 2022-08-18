import numpy as np
import pandas as pd

import utils


def parse_publish_time(publish_time: str):

    publish_time = publish_time.replace('T', ':')
    publish_time = publish_time.replace('Z', ':')
    publish_time = publish_time.replace('-', '/')

    publish_time = publish_time.split(':')

    year, month, day = publish_time[0].split('/')
    time_0, time_1, time_2 = publish_time[1:-1]

    publish_time = np.asarray([year[2:], month, day, time_0, time_1, time_2], dtype=np.float32)

    return publish_time


def parse_trending_date(trending_date: str):

    year, day, month = trending_date.split('.')

    trending_date = np.asarray([year, month, day], dtype=np.float32)

    return trending_date


def process_dates(dataframe: pd.DataFrame):

    publish_time = pd.DataFrame()
    trending_date = pd.DataFrame()

    publish_time[['year', 'month', 'day', 'time_0', 'time_1', 'time_2']] = \
        dataframe['publish_time'].apply(parse_publish_time).apply(pd.Series)

    trending_date[['trend_year', 'trend_month', 'trend_day']] = \
        dataframe['trending_date'].apply(parse_trending_date).apply(pd.Series)

    dataframe[['drift_years', 'drift_months', 'drift_days']] = \
        trending_date[['trend_year', 'trend_month', 'trend_day']].values - publish_time[['year', 'month', 'day']].values

    dataframe['trending_date'] =\
        100 * trending_date['trend_year'].values \
        + 10 * trending_date['trend_month'].values + trending_date['trend_day'].values

    return dataframe


def process_text(dataframe, column_name, exclude_set=None):

    dataframe[column_name] = dataframe[column_name].fillna('')

    (_, common_words) = utils.find_common_words(dataframe[column_name])

    if exclude_set is not None:

        common_words = common_words - exclude_set

    # empty
    if not len(common_words):

        return dataframe

    words_count = utils.bag_of_words(dataframe[column_name], words=common_words, return_counts=True)

    dataframe[f'{column_name}_size'] = dataframe[column_name].apply(lambda sstr: len(sstr))

    dataframe.drop(column_name, axis=1, inplace=True)

    # rename
    columns_map = list(map(lambda name: {name: column_name + '_has_' + name}, words_count.columns))

    for name_map in columns_map:

        words_count.rename(columns=name_map, inplace=True)

    # normalized
    words_count = np.log(words_count + 1.0)

    # concat
    dataframe = pd.concat([dataframe, words_count], axis=1)

    return dataframe


def create_processed(dataset_path):

    data = pd.read_csv(dataset_path)

    exclude_set = None

    # exclude_set = {'in', 'on', 'at', 'of', 'to', 'or', 'not',
    #                'and', 'be', 'is', 'are', 'was', 'were',
    #                'have', 'has', 'had', 'the'}

    data = process_dates(data)
    data = process_text(data, column_name='title', exclude_set=exclude_set)
    data = process_text(data, column_name='video_description', exclude_set=exclude_set)
    data = process_text(data, column_name='channel_title', exclude_set=exclude_set)
    data = process_text(data, column_name='tags', exclude_set=exclude_set)

    data.to_csv('./datasets/video_likes_processed.csv', index=False, encoding='utf-8')


def load_data(dataset_path=None, remake=False, as_numpy=False, predictors=None, include=None):

    if dataset_path is None:

        dataset_path = './datasets/video_likes.csv'

    if remake:

        create_processed(dataset_path=dataset_path)

    data = pd.read_csv('./datasets/video_likes_processed.csv')

    # sorted
    data = data.sort_values(['publish_time', 'trending_date',
                             'drift_years', 'drift_months', 'drift_days'], ascending=False)

    if as_numpy:

        data = utils.label_binarizer(data, 'category_id')

        exclude = {'publish_time', 'trending_date', 'likes', 'category_id'}

        if include is not None:

            exclude -= set(include)

        if predictors is not None:

            exclude -= set(predictors)

        # publish_time = utils.label_encoder(publish_time, columns=['year', 'month', 'day'])
        # trending_date = utils.label_encoder(trending_date, columns=['trend_year', 'trend_month', 'trend_day'])

        # feature engineering
        grouped = data.groupby('video_id')

        data['min_views'] = grouped['views'].transform('min')
        data['max_views'] = grouped['views'].transform('max')
        data['mean_views'] = grouped['views'].transform('mean')

        data['min_comments'] = grouped['comment_count'].transform('min')
        data['max_comments'] = grouped['comment_count'].transform('max')
        data['mean_comments'] = grouped['comment_count'].transform('mean')

        data.drop(['video_id'], axis=1, inplace=True)

        x = data.drop(exclude, axis=1)

        if predictors is not None:

            x = x[predictors]

        x = x.values.astype('float32')
        y = data[['likes']].values.astype('float32')

        return x, y

    return data
