import os
from openslide.deepzoom import DeepZoomGenerator


def all_path(dirname, file_type):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if file_type in apath:
                result.append(apath)
    return result


if __name__ == '__main__':
    path = r"WSI path"
    paths = all_path(path, '.mrxs')

    for WSI_path in paths:
        print('start processing')
        os.system('{} {} {}'.format('python', 'inference.py',WSI_path))
