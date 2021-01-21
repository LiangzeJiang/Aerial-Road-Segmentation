import argparse
import pickle
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils import data
from U_Net import *
from operations import *
from mask_to_submission import *
from data_loader import ImageLoader
from PIL import Image
from Metrics import *
from skimage import morphology

def run(args):
    # build dataset
    train_set = ImageLoader(args, mode='train')
    valid_set = ImageLoader(args, mode='valid')
    test_set = ImageLoader(args, mode='test')

    # build data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False)

    # build model and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = eval(args.model)().to(device)
    criterion = eval(args.loss)()
    # if use previous model parameters
    if args.trained_model:
        model.load_state_dict(torch.load(args.output_path + 'params.pkl'))
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
        # learning rate decay
        secheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.gamma, patience=5, threshold=0.001, eps=1e-07, verbose=True)
        losses = []
        F1 = []
        losses_valid = []
        F1_valid = []
        max_f1 = 0
        # start training
        for epoch in range(args.training_epochs):
            avg_cost, avg_f1 = 0, 0
            total_batch = len(train_loader)
            model.train()
            for batch_idx, (img, gt) in enumerate(train_loader):
                img = img.permute(0, 3, 1, 2).to(device)
                gt = gt.to(device)
                seg_out = model(img)
                seg_out = torch.sigmoid(seg_out).squeeze(1)  # squeeze abundant dim
                # compute loss
                loss = criterion(seg_out, gt)
                print('epoch ' + str(epoch + 1) + ' batch' + str(batch_idx + 1) + ' current loss:', loss)
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # compute average loss
                    avg_cost += loss.detach()/total_batch
                    # compute F1-score
                    curr_f1 = compute_F1(seg_out, gt, args)
                    avg_f1 += curr_f1/total_batch
                print('epoch ' + str(epoch + 1) + ' batch' + str(batch_idx + 1) + ' current F1-score:', curr_f1)
                
            secheduler.step(loss)
            losses.append(avg_cost)
            F1.append(avg_f1)
            print('---------------epoch ' + str(epoch + 1) + ' average loss:', avg_cost)
            print('---------------epoch ' + str(epoch + 1) + ' average F1-score:', avg_f1)
            
            
            # validation
            avg_cost, avg_f1 = 0, 0
            total_batch = len(valid_loader)
            model.eval()
            for batch_idx, (img, gt) in enumerate(valid_loader):
                img = img.permute(0, 3, 1, 2).to(device)
                gt = gt.to(device)
                with torch.no_grad():
                    pred = model(img)
                    pred = torch.sigmoid(pred).squeeze(1)  # squeeze abundant dim
                    loss = criterion(pred, gt)
                    print('validation batch ' + str(batch_idx + 1) + ' current loss:', loss)
                    # compute average loss
                    avg_cost += loss.detach()/total_batch
                    # compute F1-score
                    curr_f1 = compute_F1(pred, gt, args)
                    avg_f1 += curr_f1/total_batch
                    print('current F1-score:', curr_f1)
            print('validation average F1-score:', avg_f1)
            losses_valid.append(avg_cost)
            F1_valid.append(avg_f1)
            if avg_f1 > max_f1:
                # save model
                max_f1 = avg_f1
                torch.save(model.state_dict(), args.output_path + 'params.pkl')
            
            # save validation sample image
            with torch.no_grad():
                img = img[0].permute(1,2,0).cpu().detach().numpy()
                pred = pred[0].cpu().detach().numpy()
                
                bin_mask, label_mask = gen_mask_label(pred, args)
                merged_img, overlay = make_img_overlay(img, pred)
                patched_img, overlay_patch = make_img_overlay(img, bin_mask)
                
                merged_img.save(args.output_path + 'valid_merged_img.png')
                patched_img.save(args.output_path + 'valid_patched_img.png')
                overlay.save(args.output_path + 'valid_overlay.png')
            
        # save loss and F1 score
        if args.output is not None:
            pickle.dump({"train_loss": losses, "F1_score": F1},
                        open(args.output, "wb"))
            plt.plot(range(args.training_epochs), losses)
            plt.plot(range(args.training_epochs), F1)
            plt.plot(range(args.training_epochs), losses_valid)
            plt.plot(range(args.training_epochs), F1_valid)
            plt.grid(True)
            plt.savefig(args.output_path + 'train_loss.png')
        
        
    # predict on test set
    if args.test == True:
        image_filenames = []
        model.eval()
        ind = 1
        for batch_idx, img in enumerate(test_loader):
            img = img.permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                pred = model(img)
                pred = torch.sigmoid(pred).squeeze(1)  # squeeze abundant dim
            for i in range(args.batch_size):
                # post-processing
                A = pred[i].cpu().detach().numpy()
                A[A<0.5] = 0
                A = morphology.remove_small_objects(A.astype(bool), 600)
                image_filenames.append(A)
                ind += 1
                
        # save test sample image
        with torch.no_grad():
            img = img[0].permute(1,2,0).cpu().detach().numpy()
            pred = pred[0].cpu().detach().numpy()
            
            bin_mask, label_mask = gen_mask_label(pred, args)
            merged_img, overlay = make_img_overlay(img, pred)
            merged_img_bin, bin_overlay = make_img_overlay(img, bin_mask)
            merged_img.save(args.output_path + 'test_img.png')
            overlay.save(args.output_path + 'test_overlay.png')
        # pack the preidicted results and output the csv file
        result_to_submission(args.result_path, image_filenames, args)  # the second param should be a list of image mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # optimizer args
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.2, help='lr=gamma*lr')
    parser.add_argument('--beta1', type=float, default=0.9, help='first order decaying parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='second order decaying parameter')
    # model args
    parser.add_argument('--training_epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of batch sizes')
    parser.add_argument('--test', type=bool, default=True, help='If true then test and output csv file')
    parser.add_argument('--model', type=str, default='U_Net', help='choose a model') # U_Net
    parser.add_argument('--loss', type=str, default='IoULoss', help='choose a loss function')
    parser.add_argument('--trained_model', type=bool, default=True, help='use previous model parameters')
    parser.add_argument('--foreground_threshold', type=float, default=0.25,
                        help='percentage of pixels > 1 required to assign a foreground label to a patch')
    # constant args
    parser.add_argument('--train_path', type=str, default='./data/training/images/')
    parser.add_argument('--gt_path', type=str, default='./data/training/groundtruth/')
    parser.add_argument('--valid_path', type=str, default='./data/valid/images/')
    parser.add_argument('--valid_gt_path', type=str, default='./data/valid/groundtruth/')
    parser.add_argument('--test_path', type=str, default='./data/test_set_images/')
    parser.add_argument('--result_path', type=str, default='./output/my_submission.csv')
    parser.add_argument('--output_path', type=str, default='./output/')
    parser.add_argument('--output', type=str, default="./output/result.pkl", help='Output file to save training loss\
       and accuracy.')

    args = parser.parse_args()
    print(args)
    run(args)
