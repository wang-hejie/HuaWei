from typing import *
import numpy as np
import pandas as pd
import re
from gensim import models
import multiprocessing as mp
import multiprocessing.sharedctypes as sharedctypes
import ctypes


# 将文本用model方法转换为文本向量
def text_to_vector(text: str, model):
    words = re.split('[,;]', text)
    array = np.asarray([model.wv[w] for w in words], dtype='float32')
    return array.mean(axis=0)


# 返回训练好的Word2Vec模型
def make_word2vec_model(sample_list: List[str], word2vec_size: int):
    word2vec_model = models.Word2Vec(sample_list, workers=8, vector_size=word2vec_size, min_count=1, window=2)
    return word2vec_model


# word2vec类型的特征提取
def calculate_word2vec(feature_name: str, feature_dataset_name: str,
                       behavior_features, feature_dataset, word2vec_model, word2vec_size):
    if feature_dataset_name == 'video':
        id = 'video_id'
    elif feature_dataset_name == 'user':
        id = 'user_id'
    for index, row in behavior_features.iterrows():
        features_this_row = feature_dataset[feature_dataset[id] == row[id]]
        if not features_this_row.empty and \
                type(features_this_row[feature_name].iloc[0]) is not float:
            behavior_features.loc[index, feature_name] = text_to_vector(
                text=features_this_row[feature_name].iloc[0],
                model=word2vec_model)
        else:
            # 长度和word2vec_model输出的向量维度相同
            behavior_features.loc[index, feature_name] = np.array([0.0] * word2vec_size)
    return behavior_features


# 清理文本数据，将其从DataFrame中的一列，转换为包含所有取值的列表
def pd_to_list_for_text(dataframe_col):
    feature_all_list = dataframe_col.tolist()
    feature_all_list = list({tuple(re.split('[,;]', tag)) for tag in feature_all_list if type(tag) is not float})
    return feature_all_list


# 多进程word2vec特征提取
def multiprocess_word2vec_feature_extract(feature_name: str,
                                          behavior_features: pd.DataFrame,
                                          feature_dataset: pd.DataFrame,
                                          word2vec_size: int,
                                          target: str) -> pd.DataFrame:
    # 1. 准备数据
    behavior_features = behavior_features.astype('object')
    if 'user_id' in feature_dataset:
        feature_dataset_name = 'user'
    else:
        feature_dataset_name = 'video'
    print(f'正在进行特征提取，feature_name={feature_name}, feature_dataset_name={feature_dataset_name}')
    behavior_features.insert(2, feature_name, '')
    feature_all_list = pd_to_list_for_text(feature_dataset[feature_name])
    word2vec_model = make_word2vec_model(feature_all_list, word2vec_size)

    # 2. 特征提取
    process_pool = mp.Pool(mp.cpu_count())  # 创建进程池
    core_num = mp.cpu_count()
    cut_inter = behavior_features.shape[0] // (core_num - 1)
    cut_point_list = [i for i in range(0, behavior_features.shape[0], int(cut_inter))]
    if cut_point_list[-1] != (behavior_features.shape[0]):
        cut_point_list.append(behavior_features.shape[0])
    cut_start_list = cut_point_list[:-1]
    cut_end_list = cut_point_list[1:]
    process_ret_list = []
    for i in range(core_num):
        behavior_features_splited = behavior_features.iloc[cut_start_list[i]:cut_end_list[i], :]
        process_ret = process_pool.apply_async(calculate_word2vec,
                                               args=[feature_name, feature_dataset_name, behavior_features_splited,
                                                     feature_dataset, word2vec_model, word2vec_size])
        process_ret_list.append(process_ret)
    process_pool.close()  # 关闭进程池
    process_pool.join()  # 等待进程池结束
    # 删除原始的behavior_features节省内存
    behavior_features_col_name = behavior_features.columns.values
    del behavior_features

    # 3. 将结果拼装成原behavior_features表形式，多一列新特征
    # 从多进程中拿回behavior_features
    behavior_features = pd.DataFrame(columns=behavior_features_col_name)
    for process_ret in process_ret_list:
        ret = process_ret.get()
        behavior_features = pd.concat([behavior_features, ret])
    if target == 'watch':
        behavior_features['watch_label'].astype('int32')

    return behavior_features
