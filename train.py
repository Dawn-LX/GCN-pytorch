import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import pickle
from utils import load_data,preprocess_features,preprocess_adj,tuple_to_torchSparseTensor
from gcn_model import GCN

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")

print(adj.shape)  # (2708, 2708) 
adj_hat = preprocess_adj(adj)
features = preprocess_features(features)
# features[0].shape == (49216, 2) 
# features[1].sahpe == (49216,) 
# features[2] === (2708, 1433)  


# Convert to torch.Tensor
sparse_adj_hat = tuple_to_torchSparseTensor(adj_hat)
sparse_features = tuple_to_torchSparseTensor(features)

y_train = torch.FloatTensor(y_train)  # dtype = torch.float32
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)

train_mask = torch.from_numpy(train_mask)  # dtype = torch.bool
val_mask = torch.from_numpy(val_mask)
test_mask = torch.from_numpy(test_mask)

# Build model and optimizer
model = GCN(input_dim=features[2][1],out_dim=y_train.shape[1],hidden_dims=[32])
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)  #NOTE Adam performs much better than SGD when training this GCN

# train loop
iter_all = 200
val_loss_dec_max = 10
val_loss_dec_count = 0
previous_val_loss = 999.9
from tqdm import tqdm
for ii in tqdm(range(iter_all)):
    optimizer.zero_grad() # zero the gradient buffers
    # output = model(sparse_adj_hat,sparse_features,is_sparse=True)
    output = model(sparse_adj_hat,sparse_features)

    
    train_loss = model.loss(output,y_train,train_mask)
    train_acc = model.accuracy(output,y_train,train_mask)
    val_loss = model.loss(output,y_val,val_mask)
    val_acc = model.accuracy(output,y_val,val_mask)
    
    if ii%20==0:
        print("iter_{}:train_loss=={},train_acc=={}; val_loss=={},val_acc=={},dec_count={}".format(ii,train_loss.item(),train_acc.item(),val_loss.item(),val_acc.item(),val_loss_dec_count))

    if val_loss.item() >= previous_val_loss:
        val_loss_dec_count += 1
    else:
        val_loss_dec_count = 0
    previous_val_loss = val_loss.item()

    if val_loss_dec_count >= val_loss_dec_max:
        print("Early stop occurred! The validation loss does not decrease for {} consecutive epochs".format(val_loss_dec_max))
        print("current iter={}, train_loss=={},train_acc=={}; val_loss=={},val_acc=={}".format(ii,train_loss.item(),train_acc.item(),val_loss.item(),val_acc.item()))
        break


    train_loss.backward()
    optimizer.step()

torch.save(model, 'training_dir/gcn_model.pkl')