from model import *
from data_process import *

RAW_DATA_PATH = 'E:/OneDrive/学习/竞赛/华为DIGIX2021视频推荐算法/数据集/2021_3_data'
HDF5_PATH = './input_datasets.hdf'

if __name__ == '__main__':
    # 读取数据，保存成hdf5
    # HDF5_PATH = input_and_save_to_hdf5(RAW_DATA_PATH)
    # 从hdf5中读数据集
    test_data, user_features, video_features, behavior_features = read_hdf5(HDF5_PATH)
    # 若用户没有在最后一天出现，从行为数据中剔除他的样本
    behavior_features = behavior_features[behavior_features['user_id'].isin(test_data['user_id'].unique())]

    X_train_day = [(1, 13)]
    Y_train_day = [14]

    train_behavior = sliding_window_cut_data(X_train_day, Y_train_day, behavior_features)

    # 创建模型，测试是否成功
    model = MLP()
    model.predict({'user_id': np.array([10, 10]), 'video_id': np.array([10, 10])})
    model = model.cuda()

    # 制作data_loader和test_loader
    train_loader = torch.utils.data.DataLoader(
        dataset=MLPDataset(train_behavior),
        batch_size=20, shuffle=True, num_workers=5,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=MLPDataset(test_data, train=False),
        batch_size=20, shuffle=False, num_workers=5,
    )

    # 设定损失函数，优化器
    watch_loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 2, 2, 2, 2, 2, 2, 2, 2, 2]).cuda())
    share_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    watch_pred, watch_label = model_train(model=model, epoch=10, train_loader=train_loader,
                                          watch_loss_fn=watch_loss_fn,
                                          share_loss_fn=share_loss_fn,
                                          optimizer=optimizer)

    print(watch_pred.argmax(1) == watch_label)  # 查看预测结果

    model_predict(model=model, test_loader=test_loader, test_data=test_data)
