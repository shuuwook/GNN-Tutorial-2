'''
github.com/tkipf/pygcn/blob/master/pygcn/layer.py
'''

import math

import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(1, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):        
        support = input.transpose(1,2) @ self.weight
        support = torch.bmm(adj, support)

        output = support.transpose(1,2)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Gconv(nn.Module):
    def __init__(self, in_features, out_features, num_pt, K, bias=True, up_ftr=1):
        super(Gconv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pt = num_pt
        self.K = K
        self.up_ftr = up_ftr

        self.MLP = nn.Linear(num_pt*in_features, num_pt*up_ftr*out_features)
        self.GCN = GraphConvolution(out_features, out_features, bias)
    
    def sparse_topk(self, x, K, dim=-1):
        values, indices = x.topk(max(1, min(K, x.size(dim))), dim=dim, largest=False)
        # with torch.no_grad():
        #     weight_mat = torch.zeros_like(x).scatter_(dim, indices, values)
        #     weight_mat = nn.functional.softmax(weight_mat, dim=dim)
        weight_mat = torch.zeros_like(x).scatter_(dim, indices, (1/K))
        return weight_mat.to_sparse()

    def forward(self, input):
        # (B,D*N)
        feat = self.MLP(input)

        if self.up_ftr > 1:
            feat = feat.reshape(input.size(0),-1,self.num_pt*self.up_ftr) # (B,D',N')

            dist_mat = torch.norm(feat[:,:,:,None] - feat[:,:,None,:], p=2, dim=1) # (B,N',N')
            knn_mat = self.sparse_topk(dist_mat, self.K)

            feat = self.GCN(feat, knn_mat)

            feat = feat.reshape(input.size(0),-1)  # (B,D'*N')
        
        return feat

if __name__ == '__main__':
    x = torch.randn(2,5*100)
    l = Gconv(5,10,100,K=20,up_ftr=2)
    f = l(x)
    print("gcn test.")