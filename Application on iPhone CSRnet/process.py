""" Pre-process data into h5 format"""

import h5py
import scipy.io as io
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM

root = 'data/ShanghaiTech/'
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)


def gaussian_filter_density(gt):
    # Generate density map using Gaussian filter transform

    density = np.zeros(gt.shape, dtype=np.float32)

    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    # Use KDTree to find the nearest K neighbors

    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048

    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.

            # Gaussian filter

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density


def main():
    i = 0
    for img_path in tqdm(img_paths):

        # 载入mat数据 load the data in .mat file
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))

        # Read the image
        img = plt.imread(img_path)

        # Create a matrix with the same size as the picture and all 0
        k = np.zeros((img.shape[0], img.shape[1]))

        gt = mat["image_info"][0, 0][0, 0][0]

        # Generate One-hot

        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1

        # Generate the heat map
        k = gaussian_filter_density(k)

        # Save the heat map into h5 file
        file_path = img_path.replace('.jpg', '.h5').replace('images', 'ground')

        with h5py.File(file_path, 'w') as hf:
            hf['density'] = k


if __name__ == "__main__":
    main()
