import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import re
import sys
import pandas as pd
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os.path as osp
from pathlib import Path
from skimage.filters import threshold_otsu
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


#cut WSI into patches
WSI_path = sys.argv[1:][0]
print("====================================Processing {} now=========================================".format(WSI_path))
# #PAIP
# save_path = '/media/aa/16T/Liver_Data/PAIP_patches/{}'.format(WSI_path.split('/')[-1][:-4])
# patch_save_path = save_path + '/{}'.format(WSI_path.split('/')[-1][:-4])
#HCC
save_path = 'save path/{}'.format(WSI_path.split('/')[-1][:-5])
patch_save_path = save_path + '/{}'.format(WSI_path.split('/')[-1][:-5])

if not os.path.exists(patch_save_path):
    os.makedirs(patch_save_path)

with openslide.OpenSlide(WSI_path) as slide:
    dzg = DeepZoomGenerator(slide)
    print(dzg.level_count)
    print(dzg.tile_count)
    print(dzg.level_tiles)
    print((dzg.level_dimensions)[-1])

size = (int((dzg.level_dimensions)[-1][1]/150),int((dzg.level_dimensions)[-1][0]/150))
print(size)

slide_path = WSI_path
slide = openslide.open_slide(slide_path)
"Original Slide dimensions %dx%d" % slide.dimensions
thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))

img = np.array(thumbnail.convert('L')) # convert to grayscale
thresh = threshold_otsu(img)
binary = img > thresh


def find_patches_from_slide(slide_path,filter_non_tissue=True):
    with openslide.open_slide(slide_path) as slide:
        # thumbnail = slide.get_thumbnail((slide.dimensions[0] / (150*2), slide.dimensions[1] / (150*2)))
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / (150), slide.dimensions[1] / (150)))
    
    thumbnail_grey = np.array(thumbnail.convert('L')) # convert to grayscale
    thresh = threshold_otsu(thumbnail_grey)
    binary = thumbnail_grey > thresh
    
    patches = pd.DataFrame(pd.DataFrame(binary).stack())
    patches['is_tissue'] = ~patches[0]
    patches.drop(0, axis=1, inplace=True)
    patches['slide_path'] = slide_path
    samples = patches
 
    if filter_non_tissue:
        samples = samples[samples.is_tissue == True] # remove patches with no tissue
        
    samples['tile_loc'] = list(samples.index)
    samples.reset_index(inplace=True, drop=True)

    return samples 


def gen_imgs(samples, batch_size, shuffle=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        if shuffle:
            samples = samples.sample(frac=1) # shuffle samples
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
        
            images = []
            
            for _, batch_sample in batch_samples.iterrows():
                
                with openslide.open_slide(batch_sample.slide_path) as slide:
                    tiles = DeepZoomGenerator(slide, tile_size=150, overlap=0, limit_bounds=False)
                    img = tiles.get_tile(tiles.level_count-1, batch_sample.tile_loc[::-1])

                images.append(np.array(img))

            X_train = np.array(images)
            
            yield X_train

all_tissue_samples = find_patches_from_slide(WSI_path)
print('Total patches in slide: %d' % len(all_tissue_samples))


row = []
def cut2(save_path = patch_save_path,row = row):
    slide = openslide.open_slide(row[1].slide_path)
    tiles = DeepZoomGenerator(slide, tile_size=150, overlap=0, limit_bounds=False)
    loc = [0,0]
    loc[0] =  (row[1].tile_loc)[1]
    loc[1] =  (row[1].tile_loc)[0]
    img = tiles.get_tile(tiles.level_count-1, loc)

    save_name = '{}.png'.format(loc)
    save_path2 = os.path.join(save_path, save_name)
    img.save(save_path2)


num_processes = multiprocessing.cpu_count()
print(num_processes)
proccess_number = num_processes
executor = ProcessPoolExecutor(max_workers=proccess_number)
task_list = [executor.submit(cut2,patch_save_path,row)for row in all_tissue_samples.iterrows()]
executor.shutdown(wait=True)







#classify patches based on trained PaSegNet
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

data_transforms = {
    'inference': transforms.Compose([
        # transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
print("Initializing Datasets and Dataloaders...")


inference_dataset = ImageFolderWithPaths(os.path.join(save_path), data_transforms['inference'])
inference_dataloaders = torch.utils.data.DataLoader(inference_dataset, batch_size=256, shuffle=False, num_workers=20)
print(len(inference_dataloaders))


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
        # print(model_ft.classifier[1])
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        # print(model_ft.classifier[1])
        # print(model_ft.num_classes)
        model_ft.num_classes = num_classes
        # print(model_ft.num_classes)
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
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


model_dir = 'trained PaSegNet path'
save_dir = 'path of segmentation results'
models_name = ["resnext50_32x4d"]

for model_name in models_name:
    start = time.time()

    num_classes = 8
    feature_extract = False
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    #multi-GPUs
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('trained PaSegNet checkpoint'))
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2,3])
    model.eval()
    print('Load success!')

    background = np.zeros(shape=(size[0],size[1],8))
    background[:,:,1] = 1
    output_list = np.array([])
    coor = []
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(inference_dataloaders):
            # print(i)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = nn.functional.softmax(model(inputs))

            for path in paths:
                coor.append([int(s) for s in re.findall(r'\b\d+\b', path)][-2:])

            
            output_list = np.append(output_list, outputs.cpu().numpy())
            output_list = output_list.reshape(-1, 8)

    j = 0
    for k in coor:
        background[k[1]][k[0]] = output_list[j]
        j += 1
    prob_1M05 = background

    prob_save_dir = save_dir + '/' + model_name
    if not os.path.exists(prob_save_dir):
        os.makedirs(prob_save_dir)    
    np.save(prob_save_dir + '/{}.npy'.format(WSI_path.split('/')[-1][:-4]), prob_1M05)
    end = time.time()
    print("{}Process time:%.2fs".format(model_name)%(end-start))

