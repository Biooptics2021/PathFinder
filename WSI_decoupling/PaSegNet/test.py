from __future__ import print_function
from __future__ import division
from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve


import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import label_binarize

cudnn.benchmark = True


def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = predicted_labels[i] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


class PaSegDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "01_TUM":
            label = 0
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "02_EMP":
            label = 1
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "03_FIB":
            label = 2
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "04_INF":
            label = 3
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "05_NEC":
            label = 4
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "06_NOR":
            label = 5
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "07_REA":
            label = 6
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "08_STE":
            label = 7
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


def visualize_augmentations(dataset, idx=0, samples=20, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()    


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 150

    elif model_name == "resnet34":
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 150

    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 150

    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 150

    elif model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 150

    elif model_name == "resnext101_32x8d":
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 150

    elif model_name == "resnext50_32x4d":
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 150

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 150

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 150

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        print(model_ft.classifier[1])
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        print(model_ft.classifier[1])
        print(model_ft.num_classes)
        model_ft.num_classes = num_classes
        print(model_ft.num_classes)
        input_size = 150

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 150

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size





if __name__ == '__main__':
    val_path = '/media/aa/16T/Liver_paper_folder/Liver/Data/testset_PAIP.npy'
    val_images_filepaths = np.load(val_path,allow_pickle=True)
    val_transform = A.Compose(
        [
            # A.SmallestMaxSize(max_size=160),
            # A.CenterCrop(height=128, width=128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_dataset = PaSegDataset(images_filepaths=val_images_filepaths, transform=val_transform)
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    num_classes = 8
    batch_size = 512
    num_epochs = 10
    feature_extract = False
    model, input_size = initialize_model('resnext50_32x4d', num_classes, feature_extract, use_pretrained=True)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load(model_dir + '/' + model_name + '_small_20epoch/params-{}.pkl'.format(model_name)))
    model.load_state_dict(torch.load('/media/aa/16T/Liver_paper_folder/Liver/Model-His/resnext50_32x4d_151_all_40000_train/params-resnext50_32x4d.pkl'))
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2,3])
    model.eval()
    print('Load success!')
    val_set = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=30)
    nb_classes = 8
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    prob = []
    label = []

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(val_set):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = nn.functional.softmax(model(inputs))
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            for j in outputs:
                prob.append(j.tolist())
            for k in classes:
                label.append(int(k))
    prob = np.array(prob)
    label = np.array(label)
    label = label_binarize(label, classes=[0, 1, 2, 3, 4, 5, 6, 7])

    print(confusion_matrix)
    save_path = '/media/aa/16T/Liver_paper_folder/Liver/Evaluation/classification_test/PAIP'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    np.save(save_path + '/confusion_matrix.npy', confusion_matrix.numpy())
    np.save(save_path + '/prob.npy', prob)
    np.save(save_path + '/label.npy', label)

    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    print((confusion_matrix.diag()/confusion_matrix.sum(1)).mean())