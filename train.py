# -*- coding: utf-8 -*-

"""
@Time : 2021/10/19
@Author : Lenovo
@File : train
@Description : 
"""

import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_loader import Vox1_TrainDataset, Vox1_TestDataset
from models import TDNN

from hparam import hparam as hp

def train(device, model_path=None):

    checkpoint_dir = os.path.join(hp.train.logs_root,hp.train.ckpt_dir)
    log_file = os.path.join(checkpoint_dir, 'log')
    os.makedirs(checkpoint_dir, exist_ok=True)

    message = '==========configure==========\nspeech duration={}s\nepochs={},lr={},batch size={}\n' \
              'time={}'.format(hp.data.fixed_time,hp.train.epochs,hp.train.lr,hp.train.bs,time.ctime())
    with open(log_file, 'a') as f:
        f.write(message + '\n')


    train_db = Vox1_TrainDataset()
    train_loader = DataLoader(train_db, batch_size=hp.train.bs,shuffle=True, drop_last=True)
    # train_loader = DataLoader(train_db, batch_size=hp.train.bs, num_workers=3, pin_memory=True, shuffle=True, drop_last=True)
    net = TDNN()
    if model_path:
        net.load_state_dict(torch.load(model_path))
        start_epoch = hp.train.start
    else:
        start_epoch = 0
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=hp.train.lr)
    # opt = optim.SGD(net.parameters(), lr=hp.train.lr, momentum= 0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5,gamma=0.1)


    train_loss = []
    for epoch in range(start_epoch, start_epoch+hp.train.epochs):
        net.train()
        total_correct, total_loss = 0, 0
        for step_id, (x, y) in enumerate(train_loader):
            # print(step_id)
            x = x.transpose(1, 2).to(device)
            y = y.to(device)
            y_hat, _, _ = net(x)
            loss = criterion(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(loss)

            train_loss.append(loss.item())
            total_loss += loss

            pred = y_hat.argmax(dim=1)
            correct = pred.eq(y).sum().float().item()
            total_correct += correct

            if (step_id+1) % 1 == 0:
                print("Epoch{},Step[{}/{}]: loss={:.4f},acc={:.4f}, time={}".format(epoch + 1,
                                                                                step_id + 1, len(train_loader), loss,
                                                                                correct / len(y), time.ctime()))
            # time.sleep(0.5)

        total_num = len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader)
        acc = total_correct / total_num


        if (epoch + 1) % hp.train.log_epoch == 0:
            message1 = "Epoch{}, avg_loss={:.4f}, acc={:.4f}, time={}".format(epoch + 1, avg_loss, acc, time.ctime())
            print(message1)
            if checkpoint_dir is not None:
                with open(log_file, 'a') as f:
                    f.write(message1 + '\n')


        if checkpoint_dir is not None and (epoch + 1) % hp.train.ckpt_epoch == 0:
            net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(epoch + 1) + ".pth"
            ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
            torch.save(net.state_dict(), ckpt_model_path)
            net.to(device).train()

    # save model

    net.eval().cpu()
    save_model_filename = "final_epoch_" + str(epoch + 1) + ".model"
    save_model_path = os.path.join(checkpoint_dir, save_model_filename)
    torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

if __name__ == "__main__":
    device = torch.device('cuda')
    print("Training on:", device)
    # model_path = r'checkpoints/ckpt_epoch_40.pth'
    # train(device, model_path)
    train(device)