"""To do data augmentation using flip and rotation."""
from imgaug import augmenters as iaa
import imageio
import numpy as np
import imgaug as ia
import os
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
import matplotlib.image as mpimg
import torchvision.transforms.functional as tf
from torchvision import transforms


def load_image1(infilename):
    # load image Numpy
    img = mpimg.imread(infilename)
    return img

def load_image(infilename):
    # load image PIL
    img = Image.open(infilename)
    return img


def data_aug(img, gt):
    """augment data by random rotation"""
    # get segmentation map object and corresponding image
    segmap = SegmentationMapsOnImage(gt, shape=(400, 400))
    # define transformation rule
    seq = iaa.Sequential([
        iaa.Affine(  # Apply affine transformations to some images.
            rotate=(-90, 90),  # Rotate between Â±45 degrees
            mode='symmetric' # Define the method to fill the area outside the image
        )
    ])
    # augmentation
    aug_img, aug_gt = seq(image=img, segmentation_maps=segmap)
    aug_gt = 255 * aug_gt.get_arr()

    return aug_img, aug_gt


def torch_data_aug(image, mask):
    """augment data by horizontal flipping"""
    image = tf.to_tensor(image)
    mask = tf.to_tensor(mask)
    image = tf.hflip(image)
    mask = tf.hflip(mask)
    image = transforms.ToPILImage(mode='RGB')(image)
    mask = transforms.ToPILImage(mode=None)(mask)

    return image, mask


def torch_data_aug1(image, mask):
    """augment data by vertical flipping"""
    image = tf.to_tensor(image)
    mask = tf.to_tensor(mask)
    image = tf.vflip(image)
    mask = tf.vflip(mask)
    image = transforms.ToPILImage(mode='RGB')(image)
    mask = transforms.ToPILImage(mode=None)(mask)

    return image, mask


# path of original images
train_path = './data/training/images/'
gt_path = './data/training/groundtruth/'
root_dir = "data/training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
print("Loading " + str(n) + " images")
imgs = [load_image1(image_dir + files[i]) for i in range(n)]

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image1(gt_dir + files[i]) for i in range(n)]

# data augmentation
aug_total = 3*n   # number of all augmented images

print("\n"+"Generate "+str(aug_total)+" images ")
for i in range(n):
    # generate rotation images set
    aug_imgs, aug_gt = data_aug(imgs[i], gt_imgs[i])
    # save augmented images
    img_name = "./data/training/augmented_images/satImage_" + str(100+i+1).zfill(3) + ".png"
    gt_name = "./data/training/augmented_groundtruth/satImage_" + str(100+i+1).zfill(3) + ".png"
    imageio.imwrite(img_name, aug_imgs*255)
    imageio.imwrite(gt_name, aug_gt)

imgs = [load_image(image_dir + files[i]) for i in range(n)]
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
for i in range(n):
    # generate horizontal flipping image set
    aug_imgs, aug_gt = torch_data_aug(imgs[i], gt_imgs[i])
    # save augmented images
    img_name = "./data/training/augmented_images/satImage_" + str(2*100+i+1).zfill(3) + ".png"
    gt_name = "./data/training/augmented_groundtruth/satImage_" + str(2*100+i+1).zfill(3) + ".png"
    aug_imgs.save(img_name)
    aug_gt.save(gt_name)

for i in range(n):
    # generate vertical flipping image set
    aug_imgs, aug_gt = torch_data_aug1(imgs[i], gt_imgs[i])
    # save augmented images
    img_name = "./data/training/augmented_images/satImage_" + str(3*100+i+1).zfill(3) + ".png"
    gt_name = "./data/training/augmented_groundtruth/satImage_" + str(3*100+i+1).zfill(3) + ".png"
    aug_imgs.save(img_name)
    aug_gt.save(gt_name)
