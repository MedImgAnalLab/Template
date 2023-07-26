#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Template 
@File    ：train.py
@Author  ：Yu Hui
@Date    ：2023/7/26 15:18 
"""
import os.path
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
from data.CustomDataset import CustomDataset
from model.ResNet import ResNet
from utils.utils import seed_everything, mkdir

DEBUG = False

if __name__ == '__main__':
    """
    python train.py config/config.yaml
    """
    config_path = str(sys.argv[1])
    time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('Training start time: ', time_now)
    config = yaml.load(open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config['torch_version'] = torch.__version__
    name_suffix = 'None' if config['model']['weights'] is None else config['model']['weights'].replace('/', '_')
    seed_everything(config['seed'])

    if not DEBUG:
        wandb.init(
            project=config['wandb']['project'],
            config=config,
            name=config['wandb']['name'] + '_' + config['model']['name'] + '_' + config['model']['type'] + '_' +
                 name_suffix + '_' + str(config['fold']),
            tags=config['wandb']['tags'],
            notes=time_now
        )

    device = torch.device(config['device'])

    if config['model']['weights'] is None:
        model = ResNet(config['model']['type'], None)
    elif config['model']['weights'] == 'DEFAULT':
        model = ResNet(config['model']['type'], config['model']['weights'])
    else:
        model = ResNet(config['model']['type'], None)
        model.load_state_dict(torch.load(config['model']['weights'], map_location=device))

    model.to(device)
    if config['torch_version'] > '2.0.0':
        model = torch.compile(model)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(os.path.join(config['data_path'], str(config['fold'])),
                              train_transform, train=True, test=False)
    test_dataset = CustomDataset(os.path.join(config['data_path'], str(config['fold'])),
                             test_transform, train=False, test=True)

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False,
                             num_workers=config['num_workers'], pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    best_loss = 1e10
    best_metric = {
        'acc': 0.0,
        'f1': 0.0,
        'auc': 0.0,
        'spe': 0.0,
        'sen': 0.0,
    }

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        label_list = []
        pred_list = []
        for i, (img, mask, label) in enumerate(train_loader):
            img, mask, label = img.to(device), mask.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            label_list.append(label.cpu().numpy())
            pred_list.append(pred.argmax(dim=1).cpu().numpy())

        scheduler.step()
        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)
        train_acc = np.mean(label_list == pred_list)
        train_f1 = f1_score(label_list, pred_list, average='macro')
        train_auc = roc_auc_score(label_list, pred_list)
        train_spe = recall_score(label_list, pred_list, pos_label=0)
        train_sen = recall_score(label_list, pred_list, pos_label=1)
        train_loss /= len(train_loader)
        print(
            'Train Epoch: %d, Train loss: %.4f, Train acc: %.4f, Train f1: %.4f, Train auc: %.4f, Train spe: %.4f, Train sen: %.4f' % (
                epoch + 1, train_loss, train_acc, train_f1, train_auc, train_spe, train_sen), end='\n')

        model.eval()
        test_loss = 0.0
        label_list = []
        pred_list = []
        with torch.no_grad():
            for i, (img, mask, label) in enumerate(test_loader):
                img, mask, label = img.to(device), mask.to(device), label.to(device)
                pred = model(img)
                loss = criterion(pred, label)
                test_loss += loss.item()
                label_list.append(label.cpu().numpy())
                pred_list.append(pred.argmax(dim=1).cpu().numpy())

            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)

            test_acc = np.mean(label_list == pred_list)
            test_f1 = f1_score(label_list, pred_list, average='macro')
            test_auc = roc_auc_score(label_list, pred_list)
            test_spe = recall_score(label_list, pred_list, pos_label=0)
            test_sen = recall_score(label_list, pred_list, pos_label=1)
            test_loss /= len(test_loader)
            print(
                'Test Epoch: %d, Test loss: %.4f, Test acc: %.4f, Test f1: %.4f, Test auc: %.4f, Test spe: %.4f, Test sen: %.4f' % (
                    epoch + 1, test_loss, test_acc, test_f1, test_auc, test_spe, test_sen), end='\n')

            if test_f1 > best_metric['f1'] and epoch > int(config['num_epochs'] * 0.3):
                best_metric['f1'] = test_f1
                best_metric['acc'] = test_acc
                best_metric['auc'] = test_auc
                best_metric['spe'] = test_spe
                best_metric['sen'] = test_sen
                mkdir(config['save_path'])
                torch.save(model.state_dict(), os.path.join(config['save_path'], 'best_f1_model.pth'))
                print('Best f1 model saved at Epoch %d' % epoch, end='\n')

            if test_loss < best_loss and epoch > int(config['num_epochs'] * 0.3):
                best_loss = test_loss
                torch.save(model.state_dict(), os.path.join(config['save_path'], 'best_loss_model.pth'))
                print('Best loss model saved at Epoch %d' % epoch, end='\n')

            if not DEBUG:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1': train_f1,
                    'train_auc': train_auc,
                    'train_spe': train_spe,
                    'train_sen': train_sen,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'test_f1': test_f1,
                    'test_auc': test_auc,
                    'test_spe': test_spe,
                    'test_sen': test_sen,
                }, step=epoch)

    print(f'Best metric: {best_metric}')
    print('Training end time: ', time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    if not DEBUG:
        wandb.log(best_metric)
        artifact = wandb.Artifact('best_model', type='model')
        artifact.add_file(os.path.join(config['save_path'], 'best_f1_model.pth'))
        artifact.add_file(os.path.join(config['save_path'], 'best_loss_model.pth'))
        wandb.log_artifact(artifact)
