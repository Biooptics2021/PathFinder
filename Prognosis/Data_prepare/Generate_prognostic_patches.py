import os
import numpy as np
import pandas as pd
import openslide
from random import sample
from openslide.deepzoom import DeepZoomGenerator


def Seg_TUM(heatmap_npy):
    multi_classes = np.zeros((heatmap_npy.shape[0],heatmap_npy.shape[1]))

    for i in range(heatmap_npy.shape[0]):
        for j in range(heatmap_npy.shape[1]):
            multi_classes[i][j] = np.argmax(heatmap_npy[i,j,:])
    
    return multi_classes


def k_largest_index_argsort(a, k): 
    #coor of k tumor patches
    idx = np.argsort(a.ravel())[:-k-1:-1] 
    return np.column_stack(np.unravel_index(idx, a.shape))


def get_coor(WSI_heatmap_path, patch_num):
    heatmap_npy = np.load(WSI_heatmap_path)
    heatmap_npy = heatmap_npy[4:-4,4:-4,:]
    TUM = heatmap_npy[:,:,0]

    multi_classes = Seg_TUM(heatmap_npy)
    coor = np.where(multi_classes==0)
    x,y = coor
    coor_new = []

    if len(x) < patch_num:
        print('The TUM patches is not enough! Change to prob.')
        coor_new = k_largest_index_argsort(TUM, k=patch_num)
    else:
        for i in range(len(x)):
            coor_new.append([x[i], y[i]])
        coor_new = sample(coor_new, patch_num)

    return coor_new


def generate_patches_40X(WSI_path,coor,save_base_dir):
    slide = openslide.open_slide(WSI_path)
    tiles = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)

    for loc in coor:

        loc = [int(loc[1]*(150/512)+1),int(loc[0]*(150/512)+1)]
        img = tiles.get_tile(tiles.level_count-2, loc)

        save_name = '/{}.png'.format(loc)
        save_dir = save_base_dir + '/{}'.format(WSI_path.split('/')[-1][:-4])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 

        save_path = save_dir + save_name
        img.save(save_path)


def generate_patches_20X(WSI_path,coor,save_base_dir):
    slide = openslide.open_slide(WSI_path)
    tiles = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)

    for loc in coor:

        loc = [int(loc[1]*(150/512)+1),int(loc[0]*(150/512)+1)]
        img = tiles.get_tile(tiles.level_count-1, loc)

        save_name = '/{}.png'.format(loc)
        save_dir = save_base_dir + '/{}'.format(WSI_path.split('/')[-1][:-4])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 

        save_path = save_dir + save_name
        img.save(save_path)


def get_patches(WSI_path, save_base_dir, patch_num):

    WSI_heatmap_path = ''+WSI_path.split('/')[-1][:-4]+'.npy'
    coor = get_coor(WSI_heatmap_path, patch_num)
    slide = openslide.open_slide(WSI_path)
    obj_power = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    if obj_power == '40':
        generate_patches_40X(WSI_path,coor,save_base_dir)
    else:
        generate_patches_20X(WSI_path,coor,save_base_dir)
    
    print('{} is done'.format(WSI_path.split('/')[-1][:-4]))


def get_files(path, rule=".svs"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all


if __name__ == '__main__':
    path = ''#WSIs path
    paths = get_files(path)
    clinical_list = list(pd.read_csv('')['WSIs'])#clinical info

    cleaned_paths = []
    for i in range(len(paths)):
        WSI_name = paths[i].split('/')[-1][:12]
        if WSI_name in clinical_list:
            cleaned_paths.append(paths[i])

    print(len(cleaned_paths))

    patch_num = 40
    for i in cleaned_paths:
        WSI_path = i
        save_base_dir = 'xx'
        WSI_heatmap_path = 'xx'+WSI_path.split('/')[-1][:-4]+'.npy'

        if not os.path.exists(WSI_heatmap_path):
            continue

        get_patches(WSI_path, save_base_dir, patch_num)


