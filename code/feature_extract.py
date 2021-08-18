from typing import *
import numpy as np
import pandas as pd
import re
from gensim import models


# 将文本用model方法转换为文本向量
def text_to_vector(text: str, model):
    words = re.split('[,;]', text)
    array = np.asarray([model.wv[w] for w in words], dtype='float32')
    return array.mean(axis=0)


# 返回训练好的Word2Vec模型
def make_word2vec_model(sample_list: List[str]):
    word2vec_model = models.Word2Vec(sample_list, workers=8, vector_size=20, min_count=1, window=2)
    return word2vec_model


# 多线程计算video_tags
def calculate_video_tags(behavior_features, video_features, video_tags_word2vec_model, word2vec_size):
    for index, row in behavior_features.iterrows():
        video_features_this_row = video_features[video_features['video_id'] == row['video_id']]
        if not video_features_this_row.empty and \
                type(video_features_this_row['video_tags'].iloc[0]) is not float:
            behavior_features.loc[index, 'video_tags'] = text_to_vector(
                text=video_features_this_row['video_tags'].iloc[0],
                model=video_tags_word2vec_model)
        else:
            # 长度和word2vec_model输出的向量维度相同
            behavior_features.loc[index, 'video_tags'] = np.array([0.0] * word2vec_size)
    return behavior_features



