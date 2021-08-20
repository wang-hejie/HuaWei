from model import *
from data_process import *
from feature_extract import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 1000)
RAW_DATA_PATH = 'E:/OneDrive/学习/竞赛/华为DIGIX2021视频推荐算法/数据集/2021_3_data'
HDF5_PATH = '../input_datasets.hdf'

if __name__ == '__main__':
    # -------------------- 1. 读取数据  --------------------
    # 读取数据，保存成hdf5
    HDF5_PATH = input_and_save_to_hdf5(RAW_DATA_PATH)
    # 从hdf5中读数据集
    test_data, user_features, video_features, behavior_features = read_hdf5(HDF5_PATH)
    user_features.loc[user_features['gender'] == 3, 'gender'] = 2  # gender列2和3可能意义是未知和不愿意透露，故合并

    # 若用户没有在测试集中出现，从行为数据中剔除他的样本
    behavior_features = behavior_features[behavior_features['user_id'].isin(test_data['user_id'].unique())]

    print('----------1. 读取数据成功!----------')

    # -------------------- 2. 划分数据  --------------------
    X_train_day = [(1, 13)]
    Y_train_day = [14]
    train_behavior_watch = sliding_window_cut_data(X_train_day, Y_train_day, behavior_features, target='watch')
    print('=====watch_label.value_counts()=====')
    print(train_behavior_watch['watch_label'].value_counts())

    # X_train_day = [(1, 13), (1, 12), (1, 11), (1, 10), (1, 9)]
    # Y_train_day = [14, 13, 12, 11, 10]
    # train_behavior_share = sliding_window_cut_data(X_train_day, Y_train_day, behavior_features, target='share')
    # print("=====share_label.value_counts()=====")
    # print(train_behavior_share['share_label'].value_counts())
    print('----------2. 划分数据成功!----------')

    # -------------------- 3. 多进程特征提取  --------------------
    # 文本型特征
    word2vec_feature_dict = {'video': ['video_second_class', 'video_actor_list', 'video_director_list'],
                             'user': []}
    # 数值型特征
    value_feature_dict = {'video': ['video_score', 'video_duration'],
                          'user': []}
    # 数字型特征
    num_feature_dict = {'video': [],
                        'user': ['age', 'gender', 'province', 'city_level', 'device_name']}
    word2vec_feature_list = word2vec_feature_dict['video'] + word2vec_feature_dict['user']
    value_feature_list = value_feature_dict['video'] + value_feature_dict['user']
    num_feature_list = num_feature_dict['video'] + num_feature_dict['user']
    feature_list = word2vec_feature_list + value_feature_list + num_feature_list
    print(f'feature_list = {feature_list}')

    # 文本型特征
    for feature_dataset_name, feature_name_list in word2vec_feature_dict.items():
        if feature_dataset_name == 'video':
            feature_dataset = video_features
        else:
            feature_dataset = user_features
        for feature_name in feature_name_list:
            train_behavior_watch = multiprocess_word2vec_feature_extract(feature_name=feature_name,
                                                                         behavior_features=train_behavior_watch,
                                                                         feature_dataset=feature_dataset,
                                                                         word2vec_size=5,
                                                                         target='watch')
            # train_behavior_share = multiprocess_word2vec_feature_extract(feature_name=feature_name,
            #                                                              behavior_features=train_behavior_share,
            #                                                              feature_dataset=feature_dataset,
            #                                                              word2vec_size=5,
            #                                                              target='share')
            test_data = multiprocess_word2vec_feature_extract(feature_name=feature_name,
                                                              behavior_features=test_data,
                                                              feature_dataset=feature_dataset,
                                                              word2vec_size=5,
                                                              target='test')
    # 数值型特征
    for feature_dataset_name, feature_name_list in value_feature_dict.items():
        if feature_dataset_name == 'video':
            feature_dataset = video_features
            feature_dataset_id = ['video_id']
            feature_dataset_save_col = feature_dataset_id + feature_name_list
        else:
            feature_dataset = user_features
            feature_dataset_id = ['user_id']
            feature_dataset_save_col = feature_dataset_id + feature_name_list
        train_behavior_watch = pd.merge(train_behavior_watch, feature_dataset[feature_dataset_save_col],
                                        on=feature_dataset_id, how='left')
        test_data = pd.merge(test_data, feature_dataset[feature_dataset_save_col],
                             on=feature_dataset_id, how='left')
        for feature in feature_name_list:
            print(f'正在进行特征提取，feature_name={feature}, feature_dataset_name={feature_dataset_name}')
            train_behavior_watch[feature] = train_behavior_watch[feature].fillna(0)
            train_behavior_watch[feature] = (train_behavior_watch[feature] - train_behavior_watch[feature].min()) / \
                                            (train_behavior_watch[feature].max() - train_behavior_watch[feature].min())
            test_data[feature] = test_data[feature].fillna(0)
            test_data[feature] = (test_data[feature] - test_data[feature].min()) / \
                                 (test_data[feature].max() - test_data[feature].min())
    # 数字型特征
    for feature_dataset_name, feature_name_list in num_feature_dict.items():
        if feature_dataset_name == 'video':
            feature_dataset = video_features
            feature_dataset_id = ['video_id']
            feature_dataset_save_col = feature_dataset_id + feature_name_list
        else:
            feature_dataset = user_features
            feature_dataset_id = ['user_id']
            feature_dataset_save_col = feature_dataset_id + feature_name_list
        train_behavior_watch = pd.merge(train_behavior_watch, feature_dataset[feature_dataset_save_col],
                                        on=feature_dataset_id, how='left')
        test_data = pd.merge(test_data, feature_dataset[feature_dataset_save_col],
                             on=feature_dataset_id, how='left')
        for feature in feature_name_list:
            print(f'正在进行特征提取，feature_name={feature}, feature_dataset_name={feature_dataset_name}')
            train_behavior_watch[feature] = train_behavior_watch[feature].fillna(0)
            test_data[feature] = test_data[feature].fillna(0)

    print('=====train_behavior_watch=====')
    print(train_behavior_watch)
    print(train_behavior_watch.shape)
    # print('=====train_behavior_share=====')
    # print(train_behavior_share)
    # print(train_behavior_share.shape)
    print('=====test_data=====')
    print(test_data)
    print(test_data.shape)
    print('----------3. 特征提取成功!----------')

    # -------------------- 4. 创建模型和DataLoader  --------------------
    # 创建模型，测试是否成功
    model_watch = MLP(target='watch')
    # model_share = MLP(target='share')
    model_watch = model_watch.cuda()
    # model_share = model_share.cuda()

    # 制作data_loader和test_loader
    train_loader_watch = torch.utils.data.DataLoader(  # 训练时记得标target
        dataset=MLPDataset(train_behavior_watch, target='watch'),
        batch_size=1000, shuffle=True, num_workers=5,
    )
    # train_loader_share = torch.utils.data.DataLoader(
    #     dataset=MLPDataset(train_behavior_share, target='share'),
    #     batch_size=100, shuffle=True, num_workers=5,
    # )
    test_loader = torch.utils.data.DataLoader(
        dataset=MLPDataset(test_data, target='test', train=False),
        batch_size=1000, shuffle=False, num_workers=5,
    )

    # 设定损失函数，优化器
    watch_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 2, 2, 2, 2, 2, 2, 2, 2, 2]).cuda())
    # share_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]).cuda())
    optimizer = torch.optim.Adam(model_watch.parameters(), lr=0.001)
    print('----------4. 创建模型和DataLoader成功!----------')

    # -------------------- 5. 训练模型  --------------------
    # 训练watch模型
    watch_pred, watch_label = model_train(model=model_watch, epoch=120, train_loader=train_loader_watch,
                                          loss_fn=watch_loss_fn, optimizer=optimizer, target='watch')
    print(watch_pred.argmax(1) == watch_label)  # 查看第一个batch的预测结果
    # # 训练share模型
    # share_pred, share_label = model_train(model=model_share, epoch=120, train_loader=train_loader_share,
    #                                       loss_fn=share_loss_fn, optimizer=optimizer, target='share')
    # print(share_pred.argmax(1) == share_label)  # 查看第一个batch的预测结果
    print('----------5. 训练模型成功!----------')

    # -------------------- 6. 预测  --------------------
    test_data = model_predict(model=model_watch, test_loader=test_loader, test_data=test_data,
                              feature_list=feature_list, target='watch')
    # model_predict(model=model_share, test_loader=test_loader, test_data=test_data,
    #               feature_list=feature_list, target='share')
    test_data['is_share'] = 0
    test_data.to_csv('submission.csv', index=None)
    print('Save submission.csv complete!')
    print('----------6. 预测测试集成功!----------')
