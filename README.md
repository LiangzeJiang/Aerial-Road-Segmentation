# CS-433 Project2: Road Segmentation on Aerial Images
**The README outlines the requirements, project structure and the way to reproduce our final results. For more details, please check our report. Thanks.**

## Introduction
This project aims to extract road from a set of aerial images. We use U-Net, which is a popular and typical image segmentation model 
and combine it with regularization, data augmentation, post morphological block and other techniques to tackle this problem. **Finally, we achieve a 0.892 F1-score on AICrowd. Submission ID: 110060**

<div style="text-align: center">
<img src="https://github.com/LiangzeJiang/Aerial-Road-Segmentation/blob/main/data/example_img_gt.png"/>
</div>

## Requirements
Apart from the basic packages, the project is done under:
**opencv-python==4.1.2.30
sklearn==0.0
imgaug==0.4.0
scikit-image==0.16.2
torch==1.7.0**

You can install the required packages by `pip install -r requirements.txt` or `conda install requirements.txt`
 

## Project Structure
Here is an overview of the architecture of our project:
```
.
├── data                 	     # Original dataset
│   ├── training         		    # Training dataset and its groundtruth
|   ├── valid                  # validation dataset and its groundtruth
│   └── test_set_images        # Testing Images with no groundtruth
├── output                		   # Model output results
|   ├── my_submission.csv      # The best submission on AICrowd
|   ├── params.pkl             # The best model parameters
|   ├── result.pkl             # The training and validation loss 
|   └── ...                    # example of result prediction, train-validation loss and F1-score and so on
├── augmentation.py            # To do data augmentation on original dataset
├── data_loader.py             # Dataloader for the dataset
├── mask_to_submission.py      # Functions converting the model output to csv format
├── Metrics.py                 # Loss functions
├── operations.py              # Helper functions
├── run.py                     # To reproduce our best results or train from scratch
├── report.pdf               	 # Report of our project
├── U_Net.py                   # Class of U-Net model
├── requirements.txt			        # Required packages
├── colab_run.ipynb            # To excute run.py on Google Colab
└── README.md
```

## How to run
We used Google Colab, which provides a free GPU, to run our model. It is also possible to run our model with CPU, but it would be time-consuming. ****

To reproduce our final results or train from scratch：
**(You can use the data in `this repo` or from `https://drive.google.com/drive/folders/1SnIyY7poHQeg9vRNJgt9cxA3CNydLfC4?usp=sharing` and skip 1, 2 and 3)**

1. Download the orignial data and make sure they are in directory `'./data/'`

2. Split the `'./data/training'` into training set and validation set(4:1 in this project), and name them as in project structure

3. Before augmenting the training set, create folder `augmented_images` and `augmented_groundtruth` under `./data/training`, then run `augmentation.py` and find your augmented data in these two folders, and copy them to `./data/training/images` and `./data/training/groundtruth`, respectively.

4. Play with the model using `run.py`. You can either:
    1)  use our pre-trained model parameters(default) by executing `run.py` directly, **it may take a few minutes to clone the params.pkl from our observation, but you can download model parameters here `https://drive.google.com/file/d/1vMxn7I7HaEzEpsNWLJkTjskv9g7O9RkH/view?usp=sharing` if there is something wrong with params.pkl(caused by LFS)**
or 
    2) set `trained_model` argument as `False` and train the model from scratch. After training you may want to test on your parameters, then set `trained_model` as `True` and execute again to make a prediction.

## Results:

<div style="text-align: center">
<img src="https://github.com/LiangzeJiang/Aerial-Road-Segmentation/blob/main/data/example_results.png"/>
</div>
