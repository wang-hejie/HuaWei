from model import *
from data_process import *
from feature_extract import *


pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 1000)
RAW_DATA_PATH = 'E:/OneDrive/学习/竞赛/华为DIGIX2021视频推荐算法/数据集/2021_3_data'
HDF5_PATH = '../input_datasets.hdf'

if __name__ == '__main__':
    # 读取数据，保存成hdf5
    # HDF5_PATH = input_and_save_to_hdf5(RAW_DATA_PATH)
    # 从hdf5中读数据集
    test_data, user_features, video_features, behavior_features = read_hdf5(HDF5_PATH)
    # 若用户没有在最后一天出现，从行为数据中剔除他的样本
    behavior_features = behavior_features[behavior_features['user_id'].isin(test_data['user_id'].unique())]

    X_train_day = [(1, 13)]
    Y_train_day = [14]

    train_behavior_watch = sliding_window_cut_data(X_train_day, Y_train_day, behavior_features, target='watch')

    # train_behavior_share = sliding_window_cut_data(X_train_day, Y_train_day, behavior_features, target='share')
    print('----------1. Sliding window done!----------')

    # 创建模型，测试是否成功
    model_watch = MLP(target='watch')
    # model_share = MLP(target='share')
    model_watch = model_watch.cuda()
    # model_share = model_share.cuda()

    # 制作data_loader和test_loader
    process_pool = mp.Pool(mp.cpu_count())
    train_loader_watch = torch.utils.data.DataLoader(  # 训练时记得标target
        dataset=MLPDataset(train_behavior_watch, video_features, process_pool=process_pool, target='watch'),
        batch_size=1000, shuffle=True, num_workers=5,
    )

    # train_loader_share = torch.utils.data.DataLoader(
    #     dataset=MLPDataset(train_behavior_share),
    #     batch_size=100, shuffle=True, num_workers=5,
    # )
    test_loader = torch.utils.data.DataLoader(
        dataset=MLPDataset(test_data, video_features, process_pool=process_pool, train=False, target='test'),
        batch_size=1000, shuffle=False, num_workers=5,
    )
    process_pool.close()
    print('----------2. Load data done!----------')

    # 设定损失函数，优化器
    watch_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 2, 2, 2, 2, 2, 2, 2, 2, 2]).cuda())
    # share_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).cuda())
    optimizer = torch.optim.Adam(model_watch.parameters(), lr=0.005)

    # 训练watch模型
    watch_pred, watch_label = model_train(model=model_watch, epoch=60, train_loader=train_loader_watch,
                                          loss_fn=watch_loss_fn, optimizer=optimizer, target='watch')
    print(watch_pred.argmax(1) == watch_label)  # 查看第一个batch的预测结果
    print('----------3. Train model done!----------')
    # 训练share模型
    # share_pred, share_label = model_train(model=model_share, epoch=30, train_loader=train_loader_share,
    #                                       loss_fn=share_loss_fn, optimizer=optimizer, target='share')
    # print(share_pred.argmax(1) == share_label)  # 查看第一个batch的预测结果

    test_data = model_predict(model=model_watch, test_loader=test_loader, test_data=test_data, target='watch')
    # model_predict(model=model_share, test_loader=test_loader, test_data=test_data, target='share')
    test_data['is_share'] = 0
    test_data.to_csv('submission.csv', index=None)
    print('Save submission.csv complete!')
    print('----------4. Predict done!----------')

