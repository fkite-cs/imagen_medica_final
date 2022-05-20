import os
import pdb
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision

from datetime import datetime
from dataset import OurDataset
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from models import VGG19Custom, AlexNetCustom

writer = SummaryWriter("tensorboard_exps")

def create_logger(file_path):
    FORMAT = '[%(levelname)s]: %(message)s'
    file_path = '{}_{:%Y-%m-%d_%H:%M}.log'.format(file_path, datetime.now())
    
    # # crete logger
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO, format=FORMAT,
        handlers=[logging.FileHandler(file_path), logging.StreamHandler(sys.stdout)]
    )
    return logger

if __name__ == "__main__":

    # args
    n_classes = 2
    batch_size = 1
    device = "cuda"
    exps_dir = "exps"
    exp_name = "vgg19" # vgg19 o alexnet
    model = "vgg19"
    save_name = "model_state.pt"
    hm_folder = "heat_maps"

    if not os.path.exists(exps_dir):
        os.mkdir(exps_dir)
    
    exp_path = os.path.join(exps_dir, exp_name)
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    path_weights = os.path.join(exp_path, save_name)

    hm_path = os.path.join(exp_path, hm_folder)
    if not os.path.exists(hm_path):
        os.mkdir(hm_path)

    # data
    test_data = OurDataset("dataset/chest_xray/test")
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # model
    if model == "alexnet":
        m = AlexNetCustom(n_classes, save_hm=True)
    else:
        m = VGG19Custom(n_classes, save_hm=True)
    data_dict = torch.load(path_weights)
    m.load_state_dict(data_dict["best_weight"])
    m.to(device)

    ## test
    labels_test = []
    outputs_test = []
    m.eval()
    with torch.no_grad():
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = next(iter(testloader))
        inputs, labels = inputs.to(device), labels.to(device)

        # forward + backward + optimize
        output = m(inputs)
        output = F.softmax(output, dim=1).detach().cpu().numpy()[:,1] # output[:, 1]
        print("Output: ", output)

    
    for i in range(len(m.heat_maps)):
        img = m.heat_maps[i]
        img = torch.sum(img.squeeze(0), 0)
        img = img.detach().cpu().numpy()
        
        plt.imshow(img)
        # plt.show()
        plt.savefig(os.path.join(hm_path, f"{i}.png"))