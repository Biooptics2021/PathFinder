#Cut QHCG,TCGA heatmap.npy into square .npy for MacroNet training
import numpy as np
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def cut_empty(matrix_path, save_base_dir):
    matrix = np.load(matrix_path)
    empty = matrix[:,:,1]
    coor = np.where(empty<1)
    top = min(coor[0])
    bottom = max(coor[0])
    left = min(coor[1])
    right = max(coor[1])
    matrix = matrix[top:bottom,left:right]

    height, width, _ = matrix.shape
    new_size = 0

    if height >= width:
        new_size = height+8
        new_matrix = np.zeros((new_size, new_size,8))
        dealt_width = int((height-width)/2)
        new_matrix[4:-4, dealt_width+4:width+(dealt_width+4)] = matrix
    else:
        new_size = width+8
        new_matrix = np.zeros((new_size, new_size,8))
        dealt_height = int((width-height)/2)
        new_matrix[dealt_height+4:height+(dealt_height+4),4:-4] = matrix

    print(new_matrix.shape)

    if os.path.exists(save_base_dir) ==  False:
        os.makedirs(save_base_dir)
    np.save(save_base_dir + '/{}.npy'.format(matrix_path.split('/')[-1][:-5]), new_matrix)


def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all


if __name__ == '__main__':

    QHCG_path = ''
    TCGA_path = ''
    QHCG = get_files(QHCG_path)
    TCGA = get_files(TCGA_path)

    save_base_dir = ''
    num_processes = multiprocessing.cpu_count()

    proccess_number = num_processes
    executor = ProcessPoolExecutor(max_workers=proccess_number)
    task_list = [executor.submit(cut_empty, matrix_dir, save_base_dir) for matrix_dir in TCGA]
    executor.shutdown(wait=True)






