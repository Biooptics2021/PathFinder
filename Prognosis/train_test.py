import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

from data_loaders import SegHeatmapDatasetLoader#import corresponding dataloader
from Networks.Macro_networks import resnext50_32x4d, regularize_path_weights#import corresponding prognostic network
from utils import CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import pickle

BATCH_SIZE = 64
EPOCH = 100
LR = 5e-3
LAMBDA_COX = 1
LAMBDA_REG = 3e-4


def train(train_data, test_data, k_th_fold, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print(device)
    cindex_test_max = 0
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)
    
    
    model = resnext50_32x4d()
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    print("Number of Trainable Parameters: %d" % count_parameters(model))

    train_transform = A.Compose(
        [   
            A.Resize(512, 512),
            # A.RandomRotate90(p = 0.75),
            # A.Normalize(mean=(0,0,0,0,0,0,0,0), std=(0,0,0,0,0,0,0,0)),
            A.Normalize(mean=(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5), std=(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)),
            ToTensorV2(),
        ]
    )

    custom_data_loader = SegHeatmapDatasetLoader(train_data, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=BATCH_SIZE, shuffle=True,drop_last=False)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[]}}
    
    for epoch in tqdm(range(EPOCH)):

        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0
        print('train_model_before_weight')
        print(list(model.parameters())[-1])
        for batch_idx, (x_path, survtime, censor,_) in enumerate(train_loader):

            censor = censor.to(device)
            x_path = x_path.to(device).type(torch.FloatTensor)
            _, pred = model(x_path)

            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss_reg = regularize_path_weights(model=model)
            loss = LAMBDA_COX*loss_cox + LAMBDA_REG*loss_reg
            loss_epoch += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information

        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        loss_epoch /= len(train_loader.dataset)
        print(risk_pred_all)

        cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)
        grad_acc_epoch = None
        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, test_data)

        metric_logger['train']['loss'].append(loss_epoch)
        metric_logger['train']['cindex'].append(cindex_epoch)
        metric_logger['train']['pvalue'].append(pvalue_epoch)
        metric_logger['train']['surv_acc'].append(surv_acc_epoch)
        metric_logger['train']['grad_acc'].append(grad_acc_epoch)

        metric_logger['test']['loss'].append(loss_test)
        metric_logger['test']['cindex'].append(cindex_test)
        metric_logger['test']['pvalue'].append(pvalue_test)
        metric_logger['test']['surv_acc'].append(surv_acc_test)
        metric_logger['test']['grad_acc'].append(grad_acc_test)


        print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:}'.format('Train', loss_epoch, 'C-Index', cindex_epoch, 'p-value', pvalue_epoch))
        print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:}\n'.format('Test', loss_test, 'C-Index', cindex_test, 'p-value', pvalue_test))

        save_path = '/{}th'.format(k_th_fold)
        if not os.path.exists(save_path): os.makedirs(save_path)

        epoch_idx = epoch
        if cindex_test_max < cindex_test:
            cindex_test_max = cindex_test
        torch.save({
        'split':k_th_fold,
        'epoch':epoch_idx,
        'data': [train_data, test_data],
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger}, 
        save_path + '/{}.pkl'.format(epoch_idx))

        pickle.dump(pred_test, open(save_path + '/pred_test_{}.pkl'.format(k_th_fold), 'wb'))

    return model, optimizer, metric_logger


def test(model, data, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.eval()

    train_transform = A.Compose(
        [   
            A.Resize(512, 512),
            # A.RandomRotate90(p = 0.75),
            A.Normalize(mean=(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5), std=(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)),
            ToTensorV2(),
        ]
    )
    custom_data_loader = SegHeatmapDatasetLoader(data, transform=train_transform)
    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=BATCH_SIZE, shuffle=False,drop_last=False)
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (x_path, survtime, censor,_) in enumerate(test_loader):

        censor = censor.to(device)
        x_path = x_path.to(device).type(torch.FloatTensor)
        _, pred = model(x_path)

        loss_cox = CoxLoss(survtime, censor, pred, device)
        loss_reg = regularize_path_weights(model=model)
        loss = LAMBDA_COX*loss_cox + LAMBDA_REG*loss_reg
        loss_test += loss.data.item()
        gt_all = None

        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information




    # Measuring Test Loss, C-Index, P-Value

    loss_test /= len(test_loader.dataset)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    grad_acc_test = None
    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test
