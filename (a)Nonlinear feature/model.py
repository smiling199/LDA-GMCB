import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from math import sqrt
torch.backends.cudnn.enabled = False


class LayerAtt(nn.Module):
    def __init__(self, inSize, outSize, gcn_layers):
        super(LayerAtt, self).__init__()
        self.layers = gcn_layers
        self.inSize = inSize
        self.outSize = outSize
        self.q = nn.Linear(inSize, outSize)
        self.k = nn.Linear(inSize, outSize)
        self.v = nn.Linear(inSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=1)
        self.actfun2 = nn.ReLU()
        self.attcnn = nn.Conv1d(in_channels=self.layers, out_channels=1, kernel_size=1, stride=1,
                            bias=True)

    def forward(self, x):# batchsize*gcn_layers*featuresize
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        out = torch.bmm(Q, K.permute(0, 2, 1)) * self.norm
        alpha = self.actfun1(out)# according to gcn_layers
        z = torch.bmm(alpha, V)
        # cnnz = self.actfun2(z)
        cnnz = self.attcnn(z)
        # cnnz = self.actfun2(cnnz)
        finalz = cnnz.squeeze(dim=1)

        return finalz



class GraphEmbedding(nn.Module):
    def __init__(self, args):
        super(GraphEmbedding, self).__init__()
        self.args = args

        # GCN & GAT Layers for lncRNA
        self.gcn_lnc1_f = GCNConv(self.args.flnc, self.args.flnc)
        self.gcn_lnc2_f = GCNConv(self.args.flnc, self.args.flnc)
        self.gat_lnc1_f = GATConv(self.args.flnc, self.args.flnc, heads=1, concat=False, edge_dim=1)

        # GCN & GAT Layers for Disease
        self.gcn_dis1_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gcn_dis2_f = GCNConv(self.args.fdis, self.args.fdis)
        self.gat_dis1_f = GATConv(self.args.fdis, self.args.fdis, heads=1, concat=False, edge_dim=1)

        # LayerAtt for Multi-Level Feature Fusion
        self.layeratt_lnc = LayerAtt(inSize=self.args.flnc, outSize=self.args.flnc, gcn_layers=2)
        self.layeratt_dis = LayerAtt(inSize=self.args.fdis, outSize=self.args.fdis, gcn_layers=2)

    def forward(self, data):
        torch.manual_seed(1)
        x_lnc = torch.randn(self.args.lncRNA_number, self.args.flnc)
        x_dis = torch.randn(self.args.disease_number, self.args.fdis)
        # lncRNA Feature Extraction
        x_lnc_f1 = torch.relu(self.gcn_lnc1_f(x_lnc.cuda(), data['ll']['edges'].cuda(),
                            data['ll']['data_matrix'][data['ll']['edges'][0], data['ll']['edges'][1]].cuda()))
        x_lnc_att = torch.relu(self.gat_lnc1_f(x_lnc_f1, data['ll']['edges'].cuda(),
                            data['ll']['data_matrix'][data['ll']['edges'][0], data['ll']['edges'][1]].cuda()))
        x_lnc_f2 = torch.relu(self.gcn_lnc2_f(x_lnc_att, data['ll']['edges'].cuda(),
                            data['ll']['data_matrix'][data['ll']['edges'][0], data['ll']['edges'][1]].cuda()))

        # Disease Feature Extraction
        x_dis_f1 = torch.relu(self.gcn_dis1_f(x_dis.cuda(), data['dd']['edges'].cuda(),
                            data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]].cuda()))
        x_dis_att = torch.relu(self.gat_dis1_f(x_dis_f1, data['dd']['edges'].cuda(),
                            data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]].cuda()))
        x_dis_f2 = torch.relu(self.gcn_dis2_f(x_dis_att, data['dd']['edges'].cuda(),
                            data['dd']['data_matrix'][data['dd']['edges'][0], data['dd']['edges'][1]].cuda()))

        # LayerAtt Fusion
        # 将各层特征堆叠为 [batch_size, num_layers, feature_size]
        X_lnc = torch.stack([x_lnc_f1, x_lnc_f2], dim=1)  # shape: (nodes, 2, flnc)
        lnc_fea = self.layeratt_lnc(X_lnc)  # shape: (nodes, flnc)

        X_dis = torch.stack([x_dis_f1, x_dis_f2], dim=1)  # shape: (nodes, 2, fdis)
        dis_fea = self.layeratt_dis(X_dis)  # shape: (nodes, fdis)

        return lnc_fea.mm(dis_fea.t()), lnc_fea, dis_fea






