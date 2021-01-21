"""Custom dataloader."""
from torch.utils import data
from operations import *

class ImageLoader(data.Dataset):
    def __init__(self, args, mode):

        # initialize parameters
        self.mode = mode
        self.train_path = args.train_path
        self.gt_path = args.gt_path
        self.valid_path = args.valid_path
        self.valid_gt_path = args.valid_gt_path
        self.test_path = args.test_path
        self.tr_files = os.listdir(args.train_path)
        self.va_files = os.listdir(args.valid_path)
        self.va_gt_files = os.listdir(args.valid_gt_path)
        self.gt_files = os.listdir(args.gt_path)
        self.te_files = os.listdir(args.test_path)

        # print dataset information
        print('==============build ' + mode + ' dataloader==============')
        if mode == 'train':
            print('found ' + str(len(self.tr_files)) + ' images......')
        elif mode == 'test':
            print('found ' + str(len(self.te_files)) + ' images......')
        elif mode == 'valid':
            print('found ' + str(len(self.va_files)) + ' images......')

    def __getitem__(self, index):
        if self.mode == 'train':
            img = load_image(self.train_path + self.tr_files[index])
            gt = load_image(self.gt_path + self.tr_files[index])
            
            return img, gt

        elif self.mode == 'valid':
            img = load_image(self.valid_path + self.va_files[index])
            gt = load_image(self.valid_gt_path + self.va_files[index])

            return img, gt


        elif self.mode == 'test':
            te_files = os.listdir(self.test_path + 'test_' + str(index + 1))
            img = load_image(self.test_path + 'test_' + str(index + 1) + '/' + te_files[0])

            return img

    def __len__(self):
        if self.mode == 'train':
            return len(self.tr_files)
        elif self.mode == 'valid':
            return len(self.va_files)
        elif self.mode == 'test':
            return len(self.te_files)

