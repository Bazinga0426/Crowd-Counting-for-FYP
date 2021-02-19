import os
import cv2
import h5py
import scipy
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from utils_imgproc import smallize_density_map, fix_singular_shape


def gen_paths_img_dm(path_file_root='data/paths_train_val_test', dataset='A'):
    path_file_root_curr = os.path.join(path_file_root, 'paths_'+dataset)
    img_paths = []
    dm_paths = []
    paths = os.listdir(path_file_root_curr)[:2]
    for i in sorted([os.path.join(path_file_root_curr, p) for p in paths]):
        with open(i, 'r') as fin:
            img_paths.append(
                sorted(
                    [l.rstrip() for l in fin.readlines()],
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
            )
        with open(i, 'r') as fin:
            dm_paths.append(
                sorted(
                    [l.rstrip().replace('images', 'ground').replace('.jpg', '.h5') for l in fin.readlines()],
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
            )
    return img_paths, dm_paths


def gen_var_from_paths(paths, stride=1, unit_len=16):
    vars = []
    format_suffix = paths[0].split('.')[-1]
    if format_suffix == 'h5':
        for ph in paths:
            dm = h5py.File(ph, 'r')['density'].value.astype(np.float32)
            if unit_len:
                dm = fix_singular_shape(dm, unit_len=unit_len)
            dm = smallize_density_map(dm, stride=stride)
            vars.append(np.expand_dims(dm, axis=-1))
    elif format_suffix == 'jpg':
        for ph in paths:
            raw = cv2.cvtColor(cv2.imread(ph), cv2.COLOR_BGR2RGB).astype(np.float32)
            if unit_len:
                raw = fix_singular_shape(raw, unit_len=unit_len)
            vars.append(raw)
        # vars = norm_by_imagenet(vars)
    else:
        print('Format suffix is wrong.')
    return np.array(vars)


def gaussian_filter_density(gt):
    #Generates a density map using Gaussian filter transformation

    density = np.zeros(gt.shape, dtype=np.float32)

    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    # FInd out the K nearest neighbours using a KDTree

    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048

    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    # query kdtree
    distances, locations = tree.query(pts, k=4)


    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point

        #Convolve with the gaussian filter

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density