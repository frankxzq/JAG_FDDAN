from __future__ import print_function
import argparse
from ast import arg
import os
import random

from sklearn import svm
import torch
import torch.optim as optim
import utils
import basenet
import disentanglement_net
import torch.nn.functional as F
import numpy as np
import warnings
from datapre import load_data, all_data, train_test_preclass
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='CLDA HSI Classification')
parser.add_argument('--batch-size', type=int, default=36, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=8, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_epoch = args.epochs
num_k1 = 6
num_k2 = 1
num_k3 = 2
num_k4 = 4
BATCH_SIZE = args.batch_size
HalfWidth = 5
n_outputs = 128
nBand = 102
patch_size = 2 * HalfWidth + 1
CLASS_NUM = 7

#load data
data_path_s = './datasets/Pavia/paviaU.mat'
label_path_s = './datasets/Pavia/paviaU_gt_7.mat'
data_path_t = './datasets/Pavia/pavia.mat'
label_path_t = './datasets/Pavia/pavia_gt_7.mat'

# data_path_s = './datasets/CL/8chengliu_pre1.mat'
# label_path_s = './datasets/CL/8chengliu_label.mat'
# data_path_t = './datasets/CL/9chengliu_pre1.mat'
# label_path_t = './datasets/CL/9chengliu_label.mat'

source_data,source_label = load_data(data_path_s,label_path_s)
target_data,target_label = load_data(data_path_t,label_path_t)
print(source_data.shape,source_label.shape)
print(target_data.shape,target_label.shape)

nDataSet = 1#sample times

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])

#seeds = [1330, 1220, 1336, 1337, 1334, 1236, 1226, 1235, 1228, 1229]
seeds = [1336]# 1336

BATCH_SIZE = 128
acc_all_ndataset = []
clean_acc_all = []
val_all_ndataset = []
val_acc_ndataset = []
best_predict_all = 0
best_test_acc = 0
train_loss_ndataset = []
best_G,best_RandPerm,best_Row,best_Column = None,None,None,None


def discrepancy1(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def discrepancy2(out1, out2):
    return torch.mean(torch.abs(out1 - out2))

def discrepancy3(out1, out2):
    pdist = nn.PairwiseDistance(p=2)
    return pdist(out1,out2)



def train(ep, data_loader, data_loader_t):

    criterion_s = nn.CrossEntropyLoss().cuda()
    criterion_recon = nn.MSELoss()
        #nn.BCELoss() nn.MSELoss()



    for batch_idx, data in enumerate(zip(data_loader, data_loader_t)):
        G.train()
        E1.train()
        E2.train()
        E3.train()
        D.train()
        F1.train()
        F2.train()
        M.train()
        R1.train()
        R2.train()


        (data_s, label_s), (data_t, _) = data
        if args.cuda:
            data_s, label_s = data_s.cuda(), label_s.cuda()
            data_t = data_t.cuda()

        data_s = Variable(data_s)
        data_t = Variable(data_t)
        # print('data_s',data_s.shape)
        # print('data_t', data_t.shape)
        label_s = Variable(label_s)
        bs = len(label_s)
        domain_labels = torch.cat((torch.zeros(data_s.shape[0]),torch.ones(data_t.shape[0]))).long().cuda()


        labels_ci = 1/CLASS_NUM*torch.ones(BATCH_SIZE,CLASS_NUM).long().cuda()


        """step1"""
        # 训练特征提取器和分类器
        for i in range(num_k1):
            optimizer_g.zero_grad()
            optimizer_e1.zero_grad()
            optimizer_e2.zero_grad()
            optimizer_e3.zero_grad()
            optimizer_d.zero_grad()  #域鉴别器
            optimizer_f.zero_grad()  #分类器
            optimizer_m.zero_grad()
            optimizer_r.zero_grad()


            feature_mix_s = G(data_s)

            feature_di_s = E1(feature_mix_s)



            output_s1 = F1(feature_di_s)
            output_s2 = F2(feature_di_s)


            loss_cls1 = criterion_s(output_s1, label_s)
            loss_cls2 = criterion_s(output_s2, label_s)



            loss1 = loss_cls1 + loss_cls2
            #loss1 = loss_cls1 + loss_cls2 + loss_cls3 + loss_cls4

            loss1.backward()
            optimizer_g.step()
            optimizer_e1.step()
            optimizer_f.step()

        """step2"""
        for i in range(num_k2):
            optimizer_g.zero_grad()
            optimizer_e1.zero_grad()
            optimizer_e2.zero_grad()
            optimizer_e3.zero_grad()
            optimizer_d.zero_grad()  #域鉴别器
            optimizer_f.zero_grad()  #分类器
            optimizer_m.zero_grad()
            optimizer_r.zero_grad()


            feature_mix_s = G(data_s)
            feature_mix_t = G(data_t)

            feature_di_s = E1(feature_mix_s)
            feature_di_t = E1(feature_mix_t)
            feature_ds_s = E2(feature_mix_s)
            feature_ds_t = E2(feature_mix_t)

            output_domain_s1 = D(feature_di_s)
            output_domain_t1 = D(feature_di_t)
            output_domain_s2 = D(feature_ds_s)
            output_domain_t2 = D(feature_ds_t)

            loss_bce2 = nn.BCEWithLogitsLoss()(output_domain_s1, output_domain_t1.detach())
            loss_bce22 = nn.BCEWithLogitsLoss()(output_domain_s2, output_domain_t2.detach())

            loss_2 = -1 * loss_bce2 - 1*loss_bce22
            #loss_2 = -loss_bce2 - loss_bce22
            loss_2.backward()

            optimizer_d.step()


        """step3"""
        optimizer_g.zero_grad()
        optimizer_e1.zero_grad()
        optimizer_e2.zero_grad()
        optimizer_e3.zero_grad()
        optimizer_d.zero_grad()  # 域鉴别器
        optimizer_f.zero_grad()  # 分类器
        optimizer_m.zero_grad()
        optimizer_r.zero_grad()

        feature_mix_s = G(data_s)
        feature_mix_t = G(data_t)
        feature_ds_s = E2(feature_mix_s)
        feature_ds_t = E2(feature_mix_t)

        output_ds_s_D = D(feature_ds_s)
        output_ds_t_D = D(feature_ds_t)
        loss_bce3 = nn.BCEWithLogitsLoss()(output_ds_s_D, output_ds_t_D.detach())
        loss_8 = -1 * loss_bce3
        loss_8.backward()
        #optimizer_g.step()
        optimizer_e2.step()



        """step3"""

        optimizer_g.zero_grad()
        optimizer_e1.zero_grad()
        optimizer_e2.zero_grad()
        optimizer_e3.zero_grad()
        optimizer_d.zero_grad()  # 域鉴别器
        optimizer_f.zero_grad()  # 分类器
        optimizer_m.zero_grad()
        optimizer_r.zero_grad()

        feature_mix_s = G(data_s)
        feature_mix_t = G(data_t)
        feature_ci_s = E3(feature_mix_s)
        feature_ci_t = E3(feature_mix_t)


        ## 第一种
        y1_s = torch.exp(F1(feature_ci_s))
        y2_s = torch.exp(F2(feature_ci_s))
        y1_t = torch.exp(F1(feature_ci_t))
        y2_t = torch.exp(F2(feature_ci_t))

        #loss_3 = - (1 / CLASS_NUM) * torch.log(y1_s) - (1 / CLASS_NUM) * torch.log(y2_s)- (1 / CLASS_NUM) * torch.log(y1_t) - (1 / CLASS_NUM) * torch.log(y2_t)
        loss_3 = - 0.05 * (1 / CLASS_NUM) * torch.log(y1_s) - 0.05 * (1 / CLASS_NUM) * torch.log(y2_s) - 0.05 * (1 / CLASS_NUM) * torch.log(y1_t) - 0.05 * (1 / CLASS_NUM) * torch.log(y2_t)
        loss_3.backward(loss_3.clone().detach())
        optimizer_e3.step()



        # ### 第二种
        # output_ci_F1_s = F1(feature_ci_s)
        # output_ci_F2_s = F2(feature_ci_s)
        # output_ci_F1_t = F1(feature_ci_t)
        # output_ci_F2_t = F2(feature_ci_t)
        #
        # loss_3 = - torch.mean(torch.log(F.softmax(output_ci_F1_s + 1e-6, dim=-1))) - torch.mean(torch.log(F.softmax(output_ci_F2_s + 1e-6, dim=-1))) - torch.mean(torch.log(F.softmax(output_ci_F1_t + 1e-6, dim=-1))) - torch.mean(torch.log(F.softmax(output_ci_F2_t + 1e-6, dim=-1)))
        #
        # loss_3.backward()
        # optimizer_e3.step()



        '''
        ### 第三种
        output_ci_F1_t = F1(feature_ci_t)
        output_ci_F2_t = F2(feature_ci_t)
        y1 = F.softmax(output_ci_F1_t)
        y2 = F.softmax(output_ci_F2_t)


        #loss_3 = -torch.mean(labels_ci*torch.log(y1+0.000001)) - torch.mean(labels_ci*torch.log(y2+0.000001))
        #loss_3 = -torch.mean(torch.log(y1 + 0.000001)) - torch.mean(torch.log(y2 + 0.000001))
        '''

        #loss_3.backward()

        #optimizer_g.step()
        #optimizer_e3.step()


        """Step4"""

        optimizer_g.zero_grad()
        optimizer_e1.zero_grad()
        optimizer_e2.zero_grad()
        optimizer_e3.zero_grad()
        optimizer_d.zero_grad()  # 域鉴别器
        optimizer_f.zero_grad()  # 分类器
        optimizer_m.zero_grad()
        optimizer_r.zero_grad()


        feature_mix_s = G(data_s)
        feature_mix_t = G(data_t)

        feature_di_s = E1(feature_mix_s)
        feature_di_t = E1(feature_mix_t)
        feature_ci_s = E3(feature_mix_s)

        output_di_s_F1 = F1(feature_di_s)
        output_di_t_F1 = F1(feature_di_t)
        output_di_s_F2 = F2(feature_di_s)
        output_di_t_F2 = F2(feature_di_t)
        output_ci_s_F1 = F1(feature_ci_s)
        output_ci_s_F2 = F2(feature_ci_s)


        loss_cr_s = criterion_s(output_di_s_F1, label_s) + criterion_s(output_di_s_F2, label_s) + criterion_s(output_ci_s_F1, label_s) + criterion_s(output_ci_s_F2, label_s)
        loss_dis1_t = -discrepancy1(output_di_t_F1, output_di_t_F2)



        loss_4 = loss_cr_s + loss_dis1_t
        loss_4.backward()
        optimizer_f.step()

        """step5"""
        for i in range(num_k3):
            optimizer_g.zero_grad()
            optimizer_e1.zero_grad()
            optimizer_e2.zero_grad()
            optimizer_e3.zero_grad()
            optimizer_d.zero_grad()  # 域鉴别器
            optimizer_f.zero_grad()  # 分类器
            optimizer_m.zero_grad()
            optimizer_r.zero_grad()

            feature_mix_s = G(data_s)
            feature_mix_t = G(data_t)

            feature_di_s = E1(feature_mix_s)
            feature_di_t = E1(feature_mix_t)
            feature_ds_s = E2(feature_mix_s)
            feature_ds_t = E2(feature_mix_t)
            feature_ci_s = E3(feature_mix_s)
            feature_ci_t = E3(feature_mix_t)

            m_12_s = discrepancy2(M(feature_di_s), M(feature_ds_s))
            m_13_s = discrepancy2(M(feature_di_s), M(feature_ci_s))
            m_14_s = discrepancy2(M(feature_ds_s), M(feature_ci_s))
            m_12_t = discrepancy2(M(feature_di_t), M(feature_ds_t))
            m_13_t = discrepancy2(M(feature_di_t), M(feature_ci_t))
            m_14_t = discrepancy2(M(feature_ds_t), M(feature_ci_t))



            loss_mal = m_12_s+m_13_s+m_12_t+m_13_t-m_14_s-m_14_t
            loss_mal.backward()
            optimizer_m.step()


        """Step6"""
        for i in range(num_k4):
            optimizer_g.zero_grad()
            optimizer_e1.zero_grad()
            optimizer_e2.zero_grad()
            optimizer_e3.zero_grad()
            optimizer_d.zero_grad()  # 域鉴别器
            optimizer_f.zero_grad()  # 分类器
            optimizer_m.zero_grad()
            optimizer_r.zero_grad()


            feature_mix_s = G(data_s)
            feature_mix_t = G(data_t)
            feature_di_s = E1(feature_mix_s)
            feature_di_t = E1(feature_mix_t)
            feature_ds_s = E2(feature_mix_s)
            feature_ds_t = E2(feature_mix_t)
            feature_ci_s = E3(feature_mix_s)
            feature_ci_t = E3(feature_mix_t)

            output_di_s_D = D(feature_di_s)
            output_di_t_D = D(feature_di_t)
            loss_bce1 = nn.BCEWithLogitsLoss()(output_di_s_D,output_di_t_D.detach())
            loss_5 = 1*loss_bce1


            output_di_t_F1 = F1(feature_di_t)
            output_di_t_F2 = F2(feature_di_t)


            loss_6 = discrepancy1(output_di_t_F1, output_di_t_F2)


            m_12_s = discrepancy2(M(feature_di_s), M(feature_ds_s))
            m_13_s = discrepancy2(M(feature_di_s), M(feature_ci_s))
            m_14_s = discrepancy2(M(feature_ds_s), M(feature_ci_s))
            m_12_t = discrepancy2(M(feature_di_t), M(feature_ds_t))
            m_13_t = discrepancy2(M(feature_di_t), M(feature_ci_t))
            m_14_t = discrepancy2(M(feature_ds_t), M(feature_ci_t))


            loss_mal = -m_12_s-m_13_s-m_12_t-m_13_t+m_14_s+m_14_t
            loss_all = loss_5 + loss_6 + loss_mal
            loss_all.backward()
            optimizer_g.step()
            optimizer_e1.step()
            optimizer_e2.step()
            optimizer_e3.step()


        """Step7"""

        ### 重构损失

        optimizer_g.zero_grad()
        optimizer_e1.zero_grad()
        optimizer_e2.zero_grad()
        optimizer_e3.zero_grad()
        optimizer_d.zero_grad()  # 域鉴别器
        optimizer_f.zero_grad()  # 分类器
        optimizer_m.zero_grad()
        optimizer_r.zero_grad()


        feature_mix_s = G(data_s)
        feature_mix_t = G(data_t)
        feature_di_s = E1(feature_mix_s)
        feature_di_t = E1(feature_mix_t)
        feature_ds_s = E2(feature_mix_s)
        feature_ds_t = E2(feature_mix_t)
        feature_ci_s = E3(feature_mix_s)
        feature_ci_t = E3(feature_mix_t)

        feature_recon1_s = R1(torch.cat((feature_di_s, feature_ds_s), dim=1))
        feature_recon1_t = R1(torch.cat((feature_di_t, feature_ds_t), dim=1))
        feature_recon2_s = R2(torch.cat((feature_di_s, feature_ci_s), dim=1))
        feature_recon2_t = R2(torch.cat((feature_di_t, feature_ci_t), dim=1))


        recon_loss1_s = criterion_recon(feature_recon1_s, feature_mix_s)
        recon_loss1_t = criterion_recon(feature_recon1_t, feature_mix_t)
        recon_loss2_s = criterion_recon(feature_recon2_s, feature_mix_s)
        recon_loss2_t = criterion_recon(feature_recon2_t, feature_mix_t)

        loss_7 = recon_loss1_s+recon_loss1_t+recon_loss2_s+recon_loss2_t

        loss_7.backward()
        #optimizer_g.step()
        optimizer_e1.step()
        optimizer_e2.step()
        optimizer_e3.step()
        optimizer_r.step()




    print(
        'Train Ep: {} \ttrian_target_dataset:{}\t源域分类损失: {:.6f} \tloss_2: {:.6f} \tloss_4: {:.6f} \tloss_mal: {:.6f}\tm_12: {:.6f}\tm_13: {:.6f}\tloss_5: {:.6f}\tloss_6: {:.6f}\tloss_dis1_t: {:.6f}\tloss_8: {:.6f}\tloss_7: {:.6f}'.format(
            ep, len(data_loader_t.dataset), loss1.item(), loss_2.item(), loss_4.item(), loss_mal.item(), m_12_t.item(), m_13_t.item(), loss_5.item(), loss_6.item(),loss_dis1_t.item(),loss_8.item(),loss_7.item()))


def test(data_loader):

    test_pred_all = []
    test_all = []
    predict = np.array([], dtype=np.int64)
    G.eval()
    E1.eval()
    E2.eval()
    E3.eval()
    D.eval()
    F1.eval()
    F2.eval()
    M.eval()
    R1.eval()
    R2.eval()



    test_loss = 0
    correct_add = 0
    size = 0

    for batch_idx, data in enumerate(data_loader):
        img, label = data
        img, label = img.cuda(), label.cuda()
        img, label = Variable(img, volatile=True), Variable(label)
        output = G(img)
        output = E1(output)

        output1 = F1(output)
        output2 = F2(output)


        output_add = output1 + output2   # 对应位置特征相加
        pred = output_add.data.max(1)[1]
        test_loss += F.nll_loss(F.log_softmax(output1, dim=1), label, size_average=False).item()
        correct_add += pred.eq(label.data).cpu().sum()  # correct
        size += label.data.size()[0]  # total
        test_all = np.concatenate([test_all, label.data.cpu().numpy()])
        test_pred_all = np.concatenate([test_pred_all, pred.cpu().numpy()])
        predict = np.append(predict, pred.cpu().numpy())
    test_accuracy = 100. * float(correct_add) / size
    test_loss /= len(data_loader.dataset)  # loss function already averages over batch size
    print('Epoch: {:d} Test set:test loss:{:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        ep, test_loss, correct_add, size, 100. * float(correct_add) / size))

    acc[iDataSet] = 100. * float(correct_add) / size
    OA = acc
    C = metrics.confusion_matrix(test_all, test_pred_all)
    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

    k[iDataSet] = metrics.cohen_kappa_score(test_all, test_pred_all)

    return test_accuracy, predict

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)

    #np.random.seed(seeds[iDataSet])
    set_seed(seeds[iDataSet])

    # data
    train_xs, train_ys = train_test_preclass(source_data, source_label, HalfWidth, 180)
    testX, testY, G_test, RandPerm, Row, Column = all_data(target_data, target_label, HalfWidth)  # (7826,5,5,72)

    train_dataset = TensorDataset(torch.tensor(train_xs), torch.tensor(train_ys))
    train_t_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(train_t_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(train_t_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    #G = basenet.EmbeddingNetHyperX(nBand, n_outputs=n_outputs, patch_size=patch_size, n_classes=CLASS_NUM).cuda()
    G = disentanglement_net.SSFTTnet()
    E1 = disentanglement_net.Feature_disentangle1()
    E2 = disentanglement_net.Feature_disentangle1()
    E3 = disentanglement_net.Feature_disentangle1()
    D = disentanglement_net.AdversarialNetwork(64)
    F1 = basenet.ResClassifier2(num_classes=CLASS_NUM, num_unit=64, middle=32)
    F2 = basenet.ResClassifier2(num_classes=CLASS_NUM, num_unit=64, middle=32)
    M = disentanglement_net.Mixer()
    R1 = disentanglement_net.Reconstructor1()
    R2 = disentanglement_net.Reconstructor1()


    lr = args.lr
    if args.cuda:
        G.cuda()
        E1.cuda()
        E2.cuda()
        E3.cuda()
        D.cuda()
        F1.cuda()
        F2.cuda()
        M.cuda()
        R1.cuda()
        R2.cuda()


    # optimizer and loss
    optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)

    optimizer_e1 = optim.SGD(list(E1.parameters()), lr=args.lr, weight_decay=0.0005)

    optimizer_e2 = optim.SGD(list(E2.parameters()), lr=args.lr, weight_decay=0.0005)

    optimizer_e3 = optim.SGD(list(E3.parameters()), lr=args.lr, weight_decay=0.0005)

    optimizer_d = optim.SGD(list(D.parameters()), lr=args.lr, weight_decay=0.0005)

    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=args.momentum, lr=args.lr, weight_decay=0.0005)

    optimizer_m = optim.SGD(list(M.parameters()), lr=args.lr, weight_decay=0.0005)

    optimizer_r = optim.SGD(list(R1.parameters()) + list(R2.parameters()), lr=args.lr, weight_decay=0.0005)







    train_num = 20
    class_weights = None

    for ep in range(1,num_epoch+1):
        train(ep, train_loader_s, train_loader_t)

    print('-' * 100, '\nTesting')

    test_accuracy, predict = test(test_loader)

    if test_accuracy >= best_test_acc:
        best_test_acc = test_accuracy
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column = G_test, RandPerm, Row, Column
        best_iDataSet = iDataSet

    torch.save({'E3':E3.state_dict(),'E1':E1.state_dict(),'E2':E2.state_dict(), 'D':D.state_dict(),'F1':F1.state_dict()},'checkpoints/pavia/model_test'+str(iDataSet)+'.pt')


print(acc)
AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)

print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

print('classification map!!!!!')
for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[ i]]][best_Column[best_RandPerm[ i]]] = best_predict_all[i] + 1

import matplotlib.pyplot as plt
def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

###################################################
hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]

classification_map(hsi_pic[5:-5, 5:-5, :], best_G[5:-5, 5:-5], 24, "./classificationMap/FDDANPC.png")
#



