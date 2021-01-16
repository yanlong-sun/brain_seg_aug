import shutil
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from skimage.io import imread
from imageio import imwrite
import os
import numpy as np


def data_aug():
    seq = iaa.SomeOf(3, [
            iaa.Crop(px=(0, 50)),  # crop images from each side by 0 to 50px (randomly chosen)
            iaa.Affine(rotate=(-25, 25)),
            # iaa.AdditiveGaussianNoise(scale=(10, 40)),
            iaa.Sometimes(0.1,
                          iaa.GaussianBlur(sigma=(0, 1.0))
                          ),  # blur images with a sigma of 0 to 3.0
            iaa.LinearContrast((0.4, 2.5)),
            iaa.ElasticTransformation(alpha=10, sigma=3),
            iaa.Sharpen((0.0, 1.0))
            ],
        random_order=True)

    n = 1
    training_image_path = './data/train/'
    aug_image_path = './data/train_aug/'
    shutil.copytree(training_image_path, aug_image_path)
    images_list = os.listdir(training_image_path)
    num_slices = len(images_list)
    for image_name in images_list[:num_slices//2]:
        if 'mask' in image_name:
            continue
        elif 'tif' in image_name:
            unaug_slice = imread(training_image_path + image_name)
            unaug_mask = np.bool_(imread(training_image_path + image_name.split('.')[0] + '_mask.tif'))
            unaug_mask = SegmentationMapsOnImage(unaug_mask, shape=unaug_slice.shape)

            for i in range(n):
                aug_slice, aug_mask = seq(image=unaug_slice, segmentation_maps=unaug_mask)
                aug_mask = aug_mask.draw(size=aug_slice.shape[:2])[0]
                slice_path = aug_image_path + image_name.split('.')[0] + str(100+i) + '.tif'
                mask_path = aug_image_path + image_name.split('.')[0] + str(100+i) + '_mask.tif'
                imwrite(slice_path, aug_slice)
                imwrite(mask_path, aug_mask)

if __name__ == '__main__':
    data_aug()
