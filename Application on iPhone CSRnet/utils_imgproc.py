import cv2
import numpy as np


def smallize_density_map(density_map, stride=1):
    if stride > 1:
        density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int) // stride).tolist(), dtype=np.float32)
        for r in range(density_map_stride.shape[0]):
            for c in range(density_map_stride.shape[1]):
                density_map_stride[r, c] = np.sum(density_map[r * stride:(r + 1) * stride, c * stride:(c + 1) * stride])
    else:
        density_map_stride = density_map
    return density_map_stride


def norm_by_imagenet(img):
    if len(img.shape) == 3:
        img = img / 255.0
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
        return img
    elif len(img.shape) == 4 or len(img.shape) == 1:
        # In the ShanghaiTech dataset, the shape of the image is different, so array.shape is (N,), which is the case
        # of "== 1".
        imgs = []
        for im in img:
            im = im / 255.0
            im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
            im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
            im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
            imgs.append(im)
        return np.array(imgs)
    else:
        print('The shape of the input picture is wrong!.')
        return None


def generator_data(data_x, data_y, batch_size=1):
    len_train = data_x.shape[0]

    while 1:
        for idx_train in range(0, len_train, batch_size):
            x, y = data_x[idx_train:idx_train + batch_size], data_y[idx_train:idx_train + batch_size]
            x, y = image_preprocessing(
                x, y,
                flip_hor=True
            )
            yield (x, y)


def image_preprocessing(x, y, flip_hor=False, brightness_shift=False):
    xs, ys = [], []
    for idx_pro in range(x.shape[0]):  ## Separate processing for each dimension of the picture
        x_, y_ = x[idx_pro], y[idx_pro]
        # Preprocessing the shape of the picture
        if flip_hor:
            x_, y_ = flip_horizontally(x_, y_)

        x_ = norm_by_imagenet(x_)  ## Image data normalization
        xs.append(x_)
        ys.append(y_)
    xs, ys = np.array(xs), np.array(ys)
    return xs, ys


def flip_horizontally(x, y):
    to_flip = np.random.randint(0, 2)
    if to_flip:
        x, y = cv2.flip(x, 1), np.expand_dims(cv2.flip(np.squeeze(y), 1), axis=-1)
        # Assuming that the shape of y is (123,456,1), after cv2.flip, the shape of y will become (123,456).
    return x, y


def fix_singular_shape(img, unit_len=16):
    """
   Some networks (such as w-net) have N maxpooling layers and series layers at the same time,
     Therefore, if there is no fixed shape that is an integer multiple of 2 ** N, the shapes will collide.
    """
    hei_dst, wid_dst = img.shape[0] + (unit_len - img.shape[0] % unit_len), img.shape[1] + (
                unit_len - img.shape[1] % unit_len)
    if len(img.shape) == 3:
        img = cv2.resize(img, (wid_dst, hei_dst), interpolation=cv2.INTER_LANCZOS4)
    elif len(img.shape) == 2:
        GT = int(round(np.sum(img)))
        img = cv2.resize(img, (wid_dst, hei_dst), interpolation=cv2.INTER_LANCZOS4)
        img = img / (np.sum(img) / GT)
    return img
