import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

class GraphConvolution(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim,out_dim),requires_grad=True)
    def forward(self,A_hat,X):
        #GrammaNote:
            # torch.sparse.mm(mat1, mat2) 
            # mat1 (SparseTensor) : the first sparse matrix to be multiplied
            # mat2 (Tensor) : the second dense matrix to be multiplied
            # return : dense matrix
        # Here, A_hat is sparse, and X is either sparse (for input feature) or dense (for hidden feature)
        return A_hat.mm(X.mm(self.weight))


class GCN(nn.Module):
    def __init__(self,input_dim,out_dim,hidden_dims=[10]):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) + 1
        
        GraphConvLayers=[]
        dim_list = [input_dim] + hidden_dims + [out_dim]
        for ii in range(self.num_layers):
            GraphConvLayers.append(
                GraphConvolution(dim_list[ii],dim_list[ii+1])
            )
        for layer in GraphConvLayers:
            nn.init.xavier_uniform(layer.weight)
        self.GraphConvLayers =  nn.Sequential(*GraphConvLayers)

    def forward(self,A_hat,X):
        hidden = X
        for ii in range(self.num_layers):
            gcn_layer = self.GraphConvLayers[ii]
            hidden = gcn_layer(A_hat,hidden)
            if ii == self.num_layers -1 :
                hidden = F.softmax(hidden,dim=1)
                Z = hidden
            else:
                hidden = F.relu(hidden)
        
        return Z
    
    def loss(self,preds,labels,labels_mask):
        BCE_loss  = torch.nn.BCELoss(reduction='none')
        loss = BCE_loss(preds,labels)
        loss = torch.mean(loss,dim=1)   
        labels_mask =labels_mask.float()
        labels_mask /= torch.mean(labels_mask)
        loss *= labels_mask
        return torch.mean(loss)
        # Explanation for mean loss
            # e.g., the loss after torch.mean(loss,dim=1): loss ==[12,14,16,11,14,18,13], 
            # and mask == [1,1,1,0,1,0,1]  np.mean(mask)==1.4
        
            #(1) Ideally:
            # loss_ideally==[12,14,16,14,13]  # only compute the loss at the labeled position
            # np.mean(loss_ideally) == 13.8

            #(2)If we use `loss *= mask` directly, without `mask /= np.mean(mask)`, then:
            # loss *= mask --> loss==[12,14,16,0,14,0,13]
            # np.mean(loss) == 9.857142

            #(3)in practice, we do this:
            # mask /= np.mean(mask) 
            # loss *= mask --> loss == [16.8, 19.6, 22.4, 0.0, 19.6, 0.0, 18.12]
            # np.mean(loss) == 13.8
            # the accuracy is computed the similar way as mean loss
    
    def accuracy(self,preds,labels,labels_mask):
        correct_preds_all = torch.argmax(preds,dim=1).eq(torch.argmax(labels,dim=1))  # dtype = torch.uint8
        correct_preds_all = correct_preds_all.float()

        labels_mask =labels_mask.float()
        labels_mask /= torch.mean(labels_mask)
        correct_preds_all *= labels_mask
        acc = torch.mean(correct_preds_all)

        return acc
    


class GCN_2layers(nn.Module):
    """an old version, only support 2-layer GCN"""
    def __init__(self,input_dim,out_dim,hidden_dims=[10]):
        super(GCN_2layers, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims) + 1

        self.W0 = nn.Parameter(torch.FloatTensor(input_dim,hidden_dims[0]),requires_grad=True)
        self.W1 = nn.Parameter(torch.FloatTensor(hidden_dims[0],out_dim),requires_grad=True)

        nn.init.xavier_uniform(self.W0)
        nn.init.xavier_uniform(self.W1)

    def forward(self,A_hat,X,is_sparse=True):
        #TODO the code for sparse and dense is acutally the same.
        if is_sparse:
            assert A_hat.is_sparse
            assert X.is_sparse
            return self.forward_sparse(A_hat,X)
        else:
            assert not A_hat.is_sparse
            assert not X.is_sparse
            return self.forward_dense(A_hat,X)
    def forward_dense(self,A_hat,X):    
        h0 = A_hat.mm(X).mm(self.W0)   # h0 = A_hat * X * W0, A_hat.shape=[2708,2708], X.shape=[2708,1433], W0.shape=[1433,10]
        # print(h.shape)               # h0.shape=[2708,10]  ( e.g., in cora dataset)
        h0 = F.relu(h0)   
        h1 = A_hat.mm(h0).mm(self.W1)  # h1 = A_hat * h * W1, A_hat.shape=[2708,2708], h0.shape=[2708,10], W1.shape=[10,7]
                                            # h1.shape = [2708,7]
        Z = F.softmax(h1,dim=1)             # Z.shape = [2708,7]
        return Z
    
    def forward_sparse(self,A_hat,X):
        #GrammaNote:
            # torch.sparse.mm(mat1, mat2) 
            # mat1 (SparseTensor) : the first sparse matrix to be multiplied
            # mat2 (Tensor) : the second dense matrix to be multiplied
            # return : dense matrix
        h0 = A_hat.mm(X.mm(self.W0))
        h0 = F.relu(h0)
        h1 = A_hat.mm(h0.mm(self.W1))
        Z = F.softmax(h1,dim=1)
        return Z
    
    def loss(self,preds,labels,labels_mask):
        BCE_loss  = torch.nn.BCELoss(reduction='none')
        loss = BCE_loss(preds,labels)
        loss = torch.mean(loss,dim=1)   # loss.size=[2708]
        labels_mask =labels_mask.float()
        labels_mask /= torch.mean(labels_mask)
        loss *= labels_mask
        return torch.mean(loss)
    
    def accuracy(self,preds,labels,labels_mask):
        correct_preds_all = torch.argmax(preds,dim=1).eq(torch.argmax(labels,dim=1))  # dtype = torch.uint8
        correct_preds_all = correct_preds_all.float()

        labels_mask =labels_mask.float()
        labels_mask /= torch.mean(labels_mask)
        correct_preds_all *= labels_mask
        acc = torch.mean(correct_preds_all)

        return acc
    

