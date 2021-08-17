from typing import *
import pandas as pd
import glob
import gc
import numpy as np
import seaborn as sns


# 将数据按各特征的取值范围设定np.dtypes，减少内存开支
def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


# 从csv中读取数据，压缩后保存成hdf5格式
def input_and_save_to_hdf5(RAW_DATA_PATH: str) -> str:
    # 读取数据
    test_data = pd.read_csv(f'{RAW_DATA_PATH}/testdata/test.csv', sep=',')
    user_features = pd.read_csv(f'{RAW_DATA_PATH}/traindata/user_features_data/user_features_data.csv', sep='\t')
    video_features = pd.read_csv(f'{RAW_DATA_PATH}/traindata/video_features_data/video_features_data.csv', sep='\t')
    behavior_features = pd.concat([
        reduce_mem(pd.read_csv(x, sep='\t')) for x in glob.glob(f'{RAW_DATA_PATH}/traindata/history_behavior_data/*/*')
    ])
    # 按日期和uid排序
    behavior_features = behavior_features.sort_values(by=['pt_d', 'user_id'])
    # 数据压缩
    test_data = reduce_mem(test_data)
    user_features = reduce_mem(user_features)
    video_features = reduce_mem(video_features)
    behavior_features = reduce_mem(behavior_features)
    # 保存为hdf5文件，方便下次读取
    hdf5_path = '../input_datasets.hdf'
    test_data.to_hdf(hdf5_path, 'test_data')
    user_features.to_hdf(hdf5_path, 'user_features')
    video_features.to_hdf(hdf5_path, 'video_features')
    behavior_features.to_hdf(hdf5_path, 'behavior_features')
    return hdf5_path


# 从hdf5文件中读取数据集
def read_hdf5(hdf5_path: str):
    test_data = pd.read_hdf(hdf5_path, 'test_data')
    user_features = pd.read_hdf(hdf5_path, 'user_features')
    video_features = pd.read_hdf(hdf5_path, 'video_features')
    behavior_features = pd.read_hdf(hdf5_path, 'behavior_features')

    return test_data, user_features, video_features, behavior_features


# 滑动窗口法将输入的每一个窗口切好，并拼接所有窗口
def sliding_window_cut_data(X_date: List[set],
                            Y_date: List[int],
                            behavior_features: pd.DataFrame,
                            target: str) -> pd.DataFrame:
    date_list = []
    for num in range(20210419, 20210430+1):
        date_list.append(num)
    for num in range(20210501, 20210502+1):
        date_list.append(num)

    train_behavior = pd.DataFrame()
    for i in range(len(X_date)):
        Y_behavior = behavior_features[behavior_features['pt_d'] == date_list[Y_date[i] - 1]]
        X_behavior = behavior_features[behavior_features['pt_d'] != 20210502]
        # 把除第14天外的其他多余日期剔除
        for date in range(13, X_date[i][1], -1):
            X_behavior = X_behavior[X_behavior['pt_d'] != date_list[date-1]]

        # 清除样本中无用的列
        X_behavior = X_behavior.drop(columns=['is_watch', 'is_share', 'is_collect', 'is_comment', 'watch_start_time',
                                              'watch_label', 'pt_d'])
        Y_behavior = Y_behavior.drop(columns=['is_watch', 'is_collect', 'is_comment', 'watch_start_time', 'pt_d'])

        Y_behavior = Y_behavior.rename(columns={"watch_label": "label_watch", "is_share": "label_share"})
        print(Y_behavior)
        # 一条训练样本：uid, vid, label_watch, label_share
        temp_train_behavior = pd.merge(X_behavior,
                                  Y_behavior[['user_id', 'video_id', 'label_watch', 'label_share']],
                                  on=['user_id', 'video_id'], how='left')

        temp_train_behavior['label_watch'] = temp_train_behavior['label_watch'].fillna(0)
        temp_train_behavior['label_share'] = temp_train_behavior['label_share'].fillna(0)
        if target == 'watch':
            temp_train_behavior = pd.concat([
                temp_train_behavior[temp_train_behavior['label_watch'] == 0].sample(50000),
                temp_train_behavior[temp_train_behavior['label_watch'] != 0]
            ])
            print(temp_train_behavior['label_watch'].value_counts())
        elif target == 'share':
            temp_train_behavior = pd.concat([
                temp_train_behavior[temp_train_behavior['label_share'] == 0].sample(1000),
                temp_train_behavior[temp_train_behavior['label_share'] != 0]
            ])
            print(temp_train_behavior['label_share'].value_counts())
        # 一个滑动窗口结束，添加到训练集中
        train_behavior = pd.concat([train_behavior, temp_train_behavior])

    return train_behavior