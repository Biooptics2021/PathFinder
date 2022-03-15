from Micro_network import resnext50_32x4d as patchnet
from Macro_network import resnext50_32x4d as segnet
from fusion.utils import dfs_freeze
import torch 
import torch.nn as nn


def regularize_MM_weights(model, reg_type=None):
    l1_reg = None

    if model.module.__hasattr__('patchnet'):
        for W in model.module.patchnet.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('segnet'):
        for W in model.module.segnet.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)

    if model.module.__hasattr__('classifier'):
        for W in model.module.classifier.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
        
    return l1_reg


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.patchnet = patchnet()
        self.segnet = segnet()

        best_patch_ckpt = torch.load('checkpoint path', map_location=torch.device('cpu'))
        best_seg_ckpt = torch.load('checkpoint path', map_location=torch.device('cpu'))
        self.patchnet.load_state_dict(best_patch_ckpt['model_state_dict'])
        self.segnet.load_state_dict(best_seg_ckpt['model_state_dict'])
        print("Load success!")

        self.classifier = nn.Sequential(nn.Linear(64, 1024),
        nn.Dropout(0.25),
        nn.Linear(1024, 64),
        nn.Dropout(0.25),
        nn.Linear(64,32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32,1))

        dfs_freeze(self.patchnet)
        dfs_freeze(self.segnet)


    def forward(self,x_patch,x_seg):
        patch_vec, _ = self.patchnet(x_patch)
        seg_vec, _ = self.segnet(x_seg)
        features = torch.cat([patch_vec,seg_vec],dim=1)
        hazard = self.classifier(features)

        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False
