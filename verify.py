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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from data_loader import Vox1_TestDataset
from models import TDNN
from utils import test_eer, compute_eer, compute_min_dcf

from hparam import hparam as hp


def get_score(embedding1, embedding2):
    score = torch.cosine_similarity(embedding1, embedding2)
    return score

def main():

    test_db = Vox1_TestDataset()
    test_loader = DataLoader(test_db, batch_size=64, num_workers=6,pin_memory=True, shuffle=True)
    # test_loader = DataLoader(test_db, batch_size=64, shuffle=True,drop_last=True)

    net = TDNN()
    net.load_state_dict(torch.load(hp.test.model_path))
    net.eval()

    scores, labels =[], []
    for step_id, (x1, x2, label) in enumerate(test_loader):
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        _, embedding1, _ = net(x1)
        _, embedding2, _ = net(x2)
        score = get_score(embedding1, embedding2)
        scores.extend(score.detach().numpy())
        labels.extend(label.numpy())
        if (step_id+1) % 10 == 0:
            print("[{}|{}], time={}".format(step_id+1,len(test_loader),time.ctime()))

    thres,eer = test_eer(scores,labels)
    print(thres,eer)

    result = compute_eer(scores, labels)
    print(result.thresh, result.eer)

    dcf = compute_min_dcf(result.fr, result.fa)
    print("dcf=", dcf)

    plt.plot(result.fa,result.fr)
    plt.show()

if __name__ == "__main__":
    main()