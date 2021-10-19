# -*- coding: utf-8 -*-

"""
@Time : 2021/10/10
@Author : Lenovo
@File : verify
@Description : 
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import Vox1_TestDataset
from models import TDNN
from utils import compute_eer, compute_min_dcf

def testeer(scores,labels):
    target_scores = []
    nontarget_scores = []
    for i in range(len(labels)):
        if labels[i] == 1:
            target_scores.append(scores[i])
        else:
            nontarget_scores.append(scores[i])
    # 排序,从小到大排序
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    # print(target_scores)  #[0.4, 0.5, 0.7, 0.8, 0.9]
    # print(nontarget_scores)  #[0.0, 0.1, 0.2, 0.3, 0.6]

    target_size = len(target_scores)  #5
    target_position = 0
    for target_position in range(target_size):  #0-4; 0, 1
        nontarget_size = len(nontarget_scores)  #5
        nontarget_n = nontarget_size * target_position * 1.0 / target_size  # 0,1
        nontarget_position = int(nontarget_size - 1 - nontarget_n)  #4, 3
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:  #
            break
    threshold = target_scores[target_position]
    print("threshold is --> ", threshold)
    eer = target_position * 1.0 / target_size
    print("eer is --> ", eer)
    return threshold, eer

def get_score(embedding1, embedding2):
    score = torch.cosine_similarity(embedding1, embedding2)
    return score

def main():

    test_db = Vox1_TestDataset()
    # test_loader = DataLoader(test_db, batch_size=2, num_workers=6,pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=64, shuffle=True,drop_last=True)

    net = TDNN()
    model_path = r''
    net.load_state_dict(torch.load(model_path))
    net.eval()

    scores, labels =[],[]
    for step_id, (x1, x2, label) in enumerate(test_loader):
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        _, embedding1, _ = net(x1)
        _, embedding2, _ = net(x2)
        score = get_score(embedding1, embedding2)
        scores.extend(score.detach().numpy())
        labels.extend(label.detach().numpy())
        # time.sleep(0.5)
        if (step_id+1) % 10 == 0:
            print("{}|{}, time={}".format(step_id+1,len(test_loader),time.ctime()))

    result = compute_eer(scores, labels)
    print(result.thresh, result.eer)

    dcf = compute_min_dcf(result.fr, result.fa)
    print("dcf=", dcf)

if __name__ == "__main__":
    main()