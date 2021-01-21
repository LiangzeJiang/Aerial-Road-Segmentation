#!/usr/bin/env python3

import os
import torch
import numpy as np
import matplotlib.image as mpimg
import re


# assign a label to a patch
def patch_to_label(patch, args):
    df = np.mean(patch)
    if df > args.foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename, args):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, args)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames, args):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, args))


def result_to_submission_strings(im, args, index):
    """For a single result array, outputs the strings that should go into the submission file"""
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, args)
            yield("{:03d}_{}_{},{}".format(index, j, i, label))


def result_to_submission(submission_filename, image_filenames, args):
    """Converts images into a submission file"""
    # if not os.path.exists(submission_filename):
    #     os.mknod(submission_filename)
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i, fn in enumerate(image_filenames):
            f.writelines('{}\n'.format(s) for s in result_to_submission_strings(fn, args, i+1))

