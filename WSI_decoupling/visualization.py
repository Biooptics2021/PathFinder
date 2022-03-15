import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from matplotlib.colors import ListedColormap
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def walk_type(path, file_type):
    paths = glob.glob(os.path.join(path,
                                   file_type
                                   ))
    return paths


def generate_heatmap(matrix_path, save_path):
    #generate heatmaps of WSIs
    prob_matrix = np.load(matrix_path)
    print(prob_matrix.shape)
    img_save_path = save_path + '/heatmap/{}'.format(matrix_path.split('/')[-1][:-4])

    if os.path.exists(img_save_path) ==  False:
        os.makedirs(img_save_path)

    classes = [' tumor', 'empty', 'fibrosis', 'inflammation',
            'necrosis', 'normal', 'reaction', 'steatosis']

    for m in range(8):
        plt.figure(figsize=(prob_matrix.shape[1]/20, prob_matrix.shape[0]/20))
        plt.xticks(())
        plt.yticks(())
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        select_label = m
        heatmap = plt.imshow(prob_matrix[:,:,select_label], cmap = 'jet',interpolation='gaussian')
        plt.savefig(img_save_path + '/{}.png'.format(classes[select_label]),bbox_inches = 'tight', pad_inches = 0)
        plt.close()

    return None


def generate_multiclasses(matrix_path, save_path):
    #generate segmentation maps of WSIs
    prob_matrix = np.load(matrix_path)
    multi_classes = np.zeros((prob_matrix.shape[0],prob_matrix.shape[1]))
    print(multi_classes.shape)

    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            multi_classes[i][j] = np.argmax(prob_matrix[i,j,:])
            if prob_matrix[i,j,0] == 0 and multi_classes[i][j] == 0:
                multi_classes[i][j] = 1

    plt.figure(figsize=(prob_matrix.shape[0]/10,prob_matrix.shape[1]/10))
    plt.xticks(())
    plt.yticks(())
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    d = ['#EE2324','#FFFFFF','#8970B2','#A0959B','#EE939A','#6DC7B2','#A4D8EA','#F89939']
    flatui = d
    sns.set_palette(flatui)

    heatmap = multi_classes
    cmap = ListedColormap(flatui)
    heatmap = plt.imshow(heatmap, cmap = cmap, interpolation = 'nearest')
    img_save_path = save_path + '/multi_classes'

    if os.path.exists(img_save_path) ==  False:
        os.makedirs(img_save_path)
 
    plt.savefig(img_save_path + '/{}.png'.format(matrix_path.split('/')[-1][:11]), bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
    plt.close('all')
    
    return None



if __name__ == '__main__':
    matrix_path = r'heatmaps .npy path'
    save_path = 'save path'
    file_type = '*.npy'
    matrix_paths = walk_type(matrix_path, file_type)

    num_processes = multiprocessing.cpu_count()
    proccess_number = num_processes
    executor = ProcessPoolExecutor(max_workers=proccess_number)
    # task_list = [executor.submit(generate_multiclasses, matrix_dir, save_path) for matrix_dir in matrix_paths[140:210]]#if RAM limited
    task_list = [executor.submit(generate_heatmap, matrix_dir, save_path) for matrix_dir in matrix_paths]
    executor.shutdown(wait=True)




