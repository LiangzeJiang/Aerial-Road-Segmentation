"""Helper functions and operations."""
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import torchvision.transforms as transforms
from mask_to_submission import patch_to_label
from sklearn.metrics import f1_score
from PIL import Image



def compute_F1(pred, gt, args):
    """extract label list"""
    patch_pred = [img_crop(pred[i].cpu().detach().numpy(), args) for i in range(args.batch_size)]
    patch_gt = [img_crop(gt[i].cpu().detach().numpy(), args) for i in range(args.batch_size)]
    f1 = f1_score(np.array(patch_gt).ravel(), np.array(patch_pred).ravel())
    return f1


def load_image(infilename):
    """load image"""
    img = mpimg.imread(infilename)
    return img


def load_images(img_dir):
    """load images in a directory"""
    files = os.listdir(img_dir)
    n = len(files)
    print("Loading " + str() + " images......")
    imgs = [load_image(img_dir + files[i]) for i in range(n)]

    return imgs


def img_float_to_uint8(img):
    """float to uint8"""
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def make_img_overlay(img, predicted_img):
    """make prediction overlay the image."""
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.4)
    return new_img, overlay

    
    
def concatenate_images(img, gt_img):
    """Concatenate an image and its groundtruth"""
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, args, w=16, h=16):
    """extract patches from a given image"""
    list_labels = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            label = patch_to_label(im_patch, args)
            list_labels.append(label)
    return list_labels
    
    
def img_break(im, win_len=100):
    """break image into small pieces"""
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    im = im[None,...] # (1,400,400,3) or (1,400,400)
    if is_2d:
        im = im[...,None] # (1,400,400,1)
        img_pieces = im[:, 0:win_len, 0:win_len, :] # (1,100,100,1)
    else:
        img_pieces = im[:, 0:win_len, 0:win_len, :] # (1,100,100,3)
    for i in range(0, imgheight, win_len):
        for j in range(0, imgwidth, win_len):
            if i == 0 and j == 0:
                continue
            im_piece = im[:, j:j + win_len, i:i + win_len, :]
            img_pieces = np.concatenate((img_pieces, im_piece), axis=0)

    return img_pieces
    
    
def img_unbreak(img_pieces, win_len = 100, is_2d = False):
    """recover image from its pieces"""
    imgheight, imgwidth = 400, 400 
    index = 0
    if is_2d:
        img = np.zeros((imgheight, imgwidth))
    else:
        img = np.zeros((3, imgheight, imgwidth))
        
    for i in range(0, imgheight, win_len):
        for j in range(0, imgwidth, win_len):
            if is_2d:
                img[j:j + win_len, i:i + win_len] = img_pieces[index,:,:]
                index += 1
            else:
                img[:, j:j + win_len, i:i + win_len] = img_pieces[index*3:(index+1)*3,:,:]
                index += 1
    return img


def gen_mask_label(mask, args):
    """generate labeled (0-1) mask and pixel-wise mask"""
    imgwidth = mask.shape[0]
    imgheight = mask.shape[1]
    bin_mask = np.zeros((imgwidth, imgheight))
    label_mask = np.zeros((int(imgwidth/16), int(imgheight/16)))
    for i in range(0, imgheight, 16):
        for j in range(0, imgwidth, 16):
            mask_patch = mask[j:j+16, i:i+16]
            label = patch_to_label(mask_patch, args)
            bin_mask[j:j+16, i:i+16] = label
            label_mask[int(j/16),int(i/16)] = label
            
    return bin_mask, label_mask
    

def post_processing(mask, kernel_size):
    """post processing for mask(Morphological Transformations: Opening)"""
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def print_network(model):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))
