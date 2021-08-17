from typing import *
import numpy as np
import pandas as pd
import re
from gensim import models


# 将文本用model方法转换为文本向量
def text_to_vector(text: str, model):
    print(f'text = {text}')
    words = re.split('[,;]', text)
    print(f'words = {words}')
    array = np.asarray([model.wv[w] for w in words], dtype='float32')
    return array.mean(axis=0)


# 返回训练好的Word2Vec模型
def make_word2vec_model(sample_list: str):
    word2vec_model = models.Word2Vec(sample_list, workers=8, vector_size=20, min_count=1, window=2)
    return word2vec_model
