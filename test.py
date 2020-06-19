import torch 
import numpy as np
import pickle
from utils import load_data,preprocess_features,preprocess_adj,tuple_to_torchSparseTensor
from gcn_model import GCN

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")

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
model_file = 'training_dir/gcn_model.pkl'
model = torch.load(model_file)
output = model(sparse_adj_hat,sparse_features)
test_loss = model.loss(output,y_test,test_mask)
test_acc = model.accuracy(output,y_test,test_mask)
print("model_file={},test_loss={},test_acc={}".format(model_file,test_loss.item(),test_acc.item()))


