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
    batch_size = 4
    device = "cuda"
    exps_dir = "exps"
    exp_name = "vgg19"
    save_name = "model_state.pt"

    model = "vgg19"

    if not os.path.exists(exps_dir):
        os.mkdir(exps_dir)
    
    exp_path = os.path.join(exps_dir, exp_name)
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    
    path_weights = os.path.join(exp_path, save_name)

    # data
    test_data = OurDataset("dataset/chest_xray/test")
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # model
    if model == "alexnet":
        m = AlexNetCustom(n_classes)
    else:
        m = VGG19Custom(n_classes)

    data_dict = torch.load(path_weights)
    m.load_state_dict(data_dict["best_weight"])
    m.to(device)

    ## test
    labels_test = []
    outputs_test = []
    m.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            output = m(inputs)
            output = F.softmax(output, dim=1).detach().cpu().numpy()[:,1] # output[:, 1]
            for _i, _j in zip(np.array(data[1]), output):
                labels_test.append(_i) 
                outputs_test.append(0 if _j < 0.5 else 1) # (4,1)

    acc = accuracy_score(y_true=labels_test, y_pred=outputs_test)
    mc = confusion_matrix(y_true=labels_test, y_pred=outputs_test)
    fig, ax = plot_confusion_matrix(conf_mat=mc, figsize=(6, 6), cmap=plt.cm.Blues)
    # plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('Actuals', fontsize=14)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=10)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, "confusion_matrix.png"))
    plt.show()
    print(f"[Test] accuracy: {acc}")
