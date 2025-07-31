import csv
import torch
import random
from train import train
import numpy as np
  
    
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        ld_data = []
        ld_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(ld_data)
    

def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def dataset(args):
    dataset = dict()

    dataset['l_d'] = read_csv(args.dataset_path + '/l_d.csv')
    

    zero_index = []
    one_index = []
    ld_pairs = []
    for i in range(dataset['l_d'].size(0)):
        for j in range(dataset['l_d'].size(1)):
            if dataset['l_d'][i][j] < 1:
                zero_index.append([i, j, 0])
            if dataset['l_d'][i][j] >= 1:
                one_index.append([i, j, 1])
   
    ld_pairs = random.sample(zero_index, len(one_index)) + one_index

    dd_matrix = read_csv(args.dataset_path + '/d_d.csv')
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}

    ll_matrix = read_csv(args.dataset_path + '/l_l.csv')
    ll_edge_index = get_edge_index(ll_matrix)
    dataset['ll'] = {'data_matrix': ll_matrix, 'edges': ll_edge_index}

    return dataset, ld_pairs


def feature_representation(model, args, dataset):
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model = train(model, dataset, optimizer, args)
    model.eval()
    with torch.no_grad():
        score, lnc_fea, dis_fea = model(dataset)
    lnc_fea = lnc_fea.cpu().detach().numpy()
    dis_fea = dis_fea.cpu().detach().numpy()
    return score, lnc_fea, dis_fea


def new_dataset(lnc_fea, dis_fea, ld_pairs):
    unknown_pairs = []
    known_pairs = []
    
    for pair in ld_pairs:
        if pair[2] == 1:
            known_pairs.append(pair[:2])
            
        if pair[2] == 0:
            unknown_pairs.append(pair[:2])
    
    
    
    print("--------------------")
    print(lnc_fea.shape,dis_fea.shape)
    print("--------------------")
    print(len(unknown_pairs), len(known_pairs))
    
    nega_list = []
    for i in range(len(unknown_pairs)):
        nega = lnc_fea[unknown_pairs[i][0],:].tolist() + dis_fea[unknown_pairs[i][1],:].tolist()+[0,1]
        nega_list.append(nega)
        
    posi_list = []
    for j in range(len(known_pairs)):
        posi = lnc_fea[known_pairs[j][0],:].tolist() + dis_fea[known_pairs[j][1],:].tolist()+[1,0]
        posi_list.append(posi)
    
    samples = posi_list + nega_list
    
    random.shuffle(samples)
    samples = np.array(samples)
    return samples

def L_Dmatix(ld_pairs,trainindex,testindex):
    l_dmatix = np.zeros((89,190))
    for i in trainindex:
        if ld_pairs[i][2]==1:
            l_dmatix[ld_pairs[i][0]][ld_pairs[i][1]]=1
    
    
    dataset = dict()
    ld_data = []
    ld_data += [[float(i) for i in row] for row in l_dmatix]
    ld_data = torch.Tensor(ld_data)
    dataset['l_d'] = ld_data
    
    train_ld_pairs = []
    test_ld_pairs = []
    for m in trainindex:
        train_ld_pairs.append(ld_pairs[m])
    
    for n in testindex:
        test_ld_pairs.append(ld_pairs[n])



    return dataset['l_d'],train_ld_pairs,test_ld_pairs