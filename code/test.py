import numpy as np
import pandas as pd
from gensim import models

def to_text_vector(txt, model):
    '''
        将文本txt转化为文本向量
    '''
    words = txt.split(',')
    array = np.asarray([model.wv[w] for w in words if w in words],dtype='float32')
    return array.mean(axis=0)

if __name__ == '__main__':
    data = {'name': ['真相', '缉魂', '刑警', '凶杀案'],
            'year': [2012, 2222, 3333, 5555],
            'reports': [4, 3, 2, 1]}
    df = pd.DataFrame(data, index=['Cochice', 'aaa', 'bbb', 'ccc'])
    print(df)
    df['lalala'] = ''
    for index, row in df.iterrows():
        df.loc[index, 'lalala'] = 123
    print(df)


    # ## 案例
    # sentences = ["1,2,3",'3,4,1','1,4,2']
    # model = models.Word2Vec(sentences, workers=8, vector_size=20, min_count = 1, window = 2)
    # a = to_text_vector(txt="1,2,3", model=model)
    # print(a)
    # print(type(a))


