import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
import scipy.sparse as sp
import re
from feature_extract import *
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 1000)


# target = 'watch' or 'share'
def model_train(model, epoch, train_loader, loss_fn, optimizer, target: str):
    for _ in range(epoch):
        for idx, data in enumerate(train_loader):
            feed_dict = {
                'user_id': data[0].long().cuda(),
                'video_id': data[1].long().cuda(),
                'video_second_class': data[2].float().cuda(),
                'video_actor_list': data[3].float().cuda(),
                'video_director_list': data[4].float().cuda(),
                'video_score': data[5].float().cuda(),
                'video_duration': data[6].float().cuda(),
                'age': data[7].int().cuda(),
                'gender': data[8].int().cuda(),
                'province': data[9].int().cuda(),
                'city_level': data[10].int().cuda(),
                'device_name': data[11].int().cuda(),
            }

            if target == 'watch':
                watch_label = data[12].long().cuda()
                label = watch_label
            elif target == 'share':
                share_label = data[12].float().cuda()
                label = share_label
            else:
                print('Target must be watch or share!')
                exit(0)

            optimizer.zero_grad()
            pred = model(feed_dict)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            # acc = ((pred.argmax(1) == label)[label != 0]).float().sum()
            acc = (pred.argmax(1) == label).float().sum()
            acc /= (label != 0).float().sum()

            if idx % 100 == 0:
                print(f'{idx}/{len(train_loader)} \t {loss.item()}\t{acc}',
                      (pred.argmax(1) == label).float().sum())
    return pred, label


def model_predict(model, test_loader, test_data, feature_list, target: str):
    # 预测测试集
    test_watch = []
    test_share = []
    with torch.no_grad():
        for data in test_loader:
            feed_dict = {
                'user_id': data[0].long().cuda(),
                'video_id': data[1].long().cuda(),
                'video_second_class': data[2].float().cuda(),
                'video_actor_list': data[3].float().cuda(),
                'video_director_list': data[4].float().cuda(),
                'video_score': data[5].float().cuda(),
                'video_duration': data[6].float().cuda(),
                'age': data[7].int().cuda(),
                'gender': data[8].int().cuda(),
                'province': data[9].int().cuda(),
                'city_level': data[10].int().cuda(),
                'device_name': data[11].int().cuda(),
            }
            if target == 'watch':
                watch_pred = model(feed_dict)
                test_watch += list(watch_pred.argmax(1).cpu().data.numpy())
            elif target == 'share':
                share_pred = model(feed_dict)
                test_share += list((share_pred.sigmoid() > 0.8).int().cpu().data.numpy().flatten())

    # 保存预测结果
    if target == 'watch':
        test_data = test_data.drop(columns=feature_list)  # 丢弃提取的特征列
        test_data['watch_label'] = test_watch
        return test_data
    elif target == 'share':
        test_data['is_share'] = test_share
        test_data.to_csv('submission.csv', index=None)
        print('Save submission.csv complete!')


# 制作data_loader用
class MLPDataset(Dataset):
    def __init__(self, behavior_features, target, train=True):
        self.behavior_features = behavior_features
        self.train = train
        self.target = target

    def __getitem__(self, index):
        user_id = self.behavior_features.iloc[index]['user_id']
        video_id = self.behavior_features.iloc[index]['video_id']
        video_second_class = self.behavior_features.iloc[index]['video_second_class']
        video_actor_list = self.behavior_features.iloc[index]['video_actor_list']
        video_director_list = self.behavior_features.iloc[index]['video_director_list']
        video_score = self.behavior_features.iloc[index]['video_score']
        video_duration = self.behavior_features.iloc[index]['video_duration']
        age = self.behavior_features.iloc[index]['age']
        gender = self.behavior_features.iloc[index]['gender']
        province = self.behavior_features.iloc[index]['province']
        city_level = self.behavior_features.iloc[index]['city_level']
        device_name = self.behavior_features.iloc[index]['device_name']

        if self.train:
            if self.target == 'watch':
                watch_label = self.behavior_features.iloc[index]['watch_label']
                watch_label = int(watch_label)
                return user_id, video_id, video_second_class, video_actor_list, video_director_list, \
                       video_score, video_duration, age, gender, province, city_level, device_name, \
                       torch.from_numpy(np.array(watch_label))
            elif self.target == 'share':
                share_label = self.behavior_features.iloc[index]['share_label']
                share_label = int(share_label)
                return user_id, video_id, video_second_class, video_actor_list, video_director_list, \
                       video_score, video_duration, age, gender, province, city_level, device_name, \
                       torch.from_numpy(np.array([share_label]))
        else:
            return user_id, video_id, video_second_class, video_actor_list, video_director_list, \
                   video_score, video_duration, age, gender, province, city_level, device_name

    def __len__(self):
        return len(self.behavior_features)


class MLP(nn.Module):

    def __init__(self, n_users=5910799, n_items=50356, n_age=8+1, n_gender=3+1,
                 n_provinces=33+1, n_citys=8+1, n_devices=1826+1,
                 word2vec_size=5, layers=[106, 64, 32],
                 dropout=False, target='watch'):
        super().__init__()
        self.user_id_embedding = torch.nn.Embedding(n_users, 32)
        self.video_id_embedding = torch.nn.Embedding(n_items, 32)
        self.video_second_class_linear = torch.nn.Linear(word2vec_size, word2vec_size)
        self.video_actor_list_linear = torch.nn.Linear(word2vec_size, word2vec_size)
        self.video_director_list_linear = torch.nn.Linear(word2vec_size, word2vec_size)
        self.video_score_linear = torch.nn.Linear(1, 1)
        self.video_duration_linear = torch.nn.Linear(1, 1)
        self.age_embedding = torch.nn.Embedding(n_age, 5)
        self.gender_embedding = torch.nn.Embedding(n_gender, 5)
        self.province_embedding = torch.nn.Embedding(n_provinces, 5)
        self.city_level_embedding = torch.nn.Embedding(n_citys, 5)
        self.device_name_embedding = torch.nn.Embedding(n_devices, 5)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        if target == 'watch':
            self.output_layer = torch.nn.Linear(layers[-1], 10)
        elif target == 'share':
            self.output_layer = torch.nn.Linear(layers[-1], 1)

    # 正向传播
    def forward(self, feed_dict):
        user_id = feed_dict['user_id']
        video_id = feed_dict['video_id']
        video_second_class = feed_dict['video_second_class']
        video_actor_list = feed_dict['video_actor_list']
        video_director_list = feed_dict['video_director_list']
        video_score = feed_dict['video_score'].reshape(-1, 1)
        video_duration = feed_dict['video_duration'].reshape(-1, 1)
        age = feed_dict['age']
        gender = feed_dict['gender']
        province = feed_dict['province']
        city_level = feed_dict['city_level']
        device_name = feed_dict['device_name']

        user_embedding_vec = self.user_id_embedding(user_id)
        video_embedding_vec = self.video_id_embedding(video_id)
        video_second_class_linear_vec = self.video_second_class_linear(video_second_class)
        video_actor_list_linear_vec = self.video_actor_list_linear(video_actor_list)
        video_director_list_linear_vec = self.video_director_list_linear(video_director_list)
        video_score_linear_vec = self.video_score_linear(video_score)
        video_duration_linear_vec = self.video_duration_linear(video_duration)
        age_embedding_vec = self.age_embedding(age)
        gender_embedding_vec = self.gender_embedding(gender)
        province_embedding_vec = self.province_embedding(province)
        city_level_embedding_vec = self.city_level_embedding(city_level)
        device_name_embedding_vec = self.device_name_embedding(device_name)

        x = torch.cat([user_embedding_vec, video_embedding_vec, video_second_class_linear_vec,
                       video_actor_list_linear_vec, video_director_list_linear_vec, video_score_linear_vec,
                       video_duration_linear_vec, age_embedding_vec, gender_embedding_vec, province_embedding_vec,
                       city_level_embedding_vec, device_name_embedding_vec], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x)
        # logit1 = self.output_layer1(x)
        logit = self.output_layer(x)
        return logit

    def predict(self, feed_dict):
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                if key in ['video_second_class', 'video_actor_list', 'video_director_list',
                           'video_score', 'video_duration']:
                    feed_dict[key] = torch.from_numpy(
                        feed_dict[key]).to(dtype=torch.float32, device='cpu')
                else:
                    feed_dict[key] = torch.from_numpy(
                        feed_dict[key]).to(dtype=torch.long, device='cpu')
        output_scores = self.forward(feed_dict)
        return output_scores

# class NeuMF(nn.Module):
#     def __init__(self, num_users=5910794, num_items=50352):
#         super(NeuMF, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.factor_num_mf = 32
#         self.factor_num_mlp =  int(64/2)
#         self.layers = [64,32,16,8]
#         self.dropout = 0.2
#
#         self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
#         self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)
#
#         self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
#         self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)
#
#         self.fc_layers = nn.ModuleList()
#         for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
#             self.fc_layers.append(torch.nn.Linear(in_size, out_size))
#             self.fc_layers.append(nn.ReLU())
#
#         self.output_layer1 = torch.nn.Linear(self.layers[-1]+self.factor_num_mf, 10)
#         self.output_layer2 = torch.nn.Linear(self.layers[-1]+self.factor_num_mf, 1)
#         self.init_weight()
#
#     def init_weight(self):
#         nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
#         nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
#         nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
#         nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
#
#         for m in self.fc_layers:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#
#         for m in self.modules():
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, feed_dict):
#         users = feed_dict['user_id']
#         items = feed_dict['video_id']
#         user_embedding_mlp = self.embedding_user_mlp(users)
#         item_embedding_mlp = self.embedding_item_mlp(items)
#
#         user_embedding_mf = self.embedding_user_mf(users)
#         item_embedding_mf = self.embedding_item_mf(items)
#
#         mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
#         mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)
#
#         for idx, _ in enumerate(range(len(self.fc_layers))):
#             mlp_vector = self.fc_layers[idx](mlp_vector)
#             mlp_vector = F.dropout(mlp_vector)
#
#         x = torch.cat([mlp_vector, mf_vector], dim=1)
#         logit1 = self.output_layer1(x)
#         logit2 = self.output_layer2(x)
#         return logit1, logit2
#
#     def predict(self, feed_dict):
#         for key in feed_dict:
#             if type(feed_dict[key]) != type(None):
#                 feed_dict[key] = torch.from_numpy(
#                     feed_dict[key]).to(dtype=torch.long, device='cpu')
#         output_scores = self.forward(feed_dict)
#         return output_scores
