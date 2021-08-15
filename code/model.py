import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


# 制作data_loader用
class MLPDataset(Dataset):
    def __init__(self, history_behavior, train=True):
        self.history_behavior = history_behavior
        self.train = train

    def __getitem__(self, index):
        user_id = self.history_behavior.iloc[index]['user_id']
        video_id = self.history_behavior.iloc[index]['video_id']

        if self.train:
            watch_label = self.history_behavior.iloc[index]['label_watch']
            share_label = self.history_behavior.iloc[index]['label_share']

            watch_label = int(watch_label)
            share_label = int(share_label)

            return user_id, video_id, \
                   torch.from_numpy(np.array(watch_label)), \
                   torch.from_numpy(np.array([share_label]))
        else:
            return user_id, video_id

    def __len__(self):
        return len(self.history_behavior)


# target = 'watch' or 'share'
def model_train(model, epoch, train_loader, loss_fn, optimizer, target: str):
    for _ in range(epoch):
        for idx, data in enumerate(train_loader):
            feed_dict = {
                'user_id': data[0].long().cuda(),
                'video_id': data[1].long().cuda(),
            }

            if target == 'watch':
                watch_label = data[2].long().cuda()
                label = watch_label
            elif target == 'share':
                share_label = data[3].float().cuda()
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


def model_predict(model, test_loader, test_data, target: str):
    # 预测测试集
    test_watch = []
    test_share = []
    with torch.no_grad():
        for data in test_loader:
            feed_dict = {
                'user_id': data[0].long().cuda(),
                'video_id': data[1].long().cuda(),
            }
            if target == 'watch':
                watch_pred = model(feed_dict)
                test_watch += list(watch_pred.argmax(1).cpu().data.numpy())
            elif target == 'share':
                share_pred = model(feed_dict)
                test_share += list((share_pred.sigmoid() > 0.5).int().cpu().data.numpy().flatten())

    # 保存预测结果
    if target == 'watch':
        test_data['watch_label'] = test_watch
        return test_data
    elif target == 'share':
        test_data['is_share'] = test_share
        test_data.to_csv('submission.csv', index=None)
        print('Save submission.csv complete!')


class MLP(nn.Module):

    def __init__(self, n_users=5910794, n_items=50352, layers=[64, 32], dropout=False, target='watch'):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(n_users, 32)
        self.video_embedding = torch.nn.Embedding(n_items, 32)

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
        users = feed_dict['user_id']
        items = feed_dict['video_id']

        user_embedding = self.user_embedding(users)
        video_embedding = self.video_embedding(items)

        x = torch.cat([user_embedding, video_embedding], 1)
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
