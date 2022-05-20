import os
import pdb
from pydoc import pathdirs
import sys
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision

from datetime import datetime
from dataset import OurDataset
from sklearn.metrics import confusion_matrix, accuracy_score
from models import VGG19Custom, AlexNetCustom

writer = SummaryWriter("tensorboard_exps")

def create_logger(file_path):
    FORMAT = '[%(levelname)s]: %(message)s'
    file_path = '{}_{:%Y-%m-%d_%H:%M}.log'.format(file_path, datetime.now())
    
    # # crete logger
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.DEBUG, format=FORMAT,
        handlers=[logging.FileHandler(file_path), logging.StreamHandler(sys.stdout)]
    )
    return logger


if __name__ == "__main__":

    # args
    lr = 0.001
    momentum = 0.9
    n_classes = 2
    print_args = 50
    do_val = 1000
    batch_size = 4
    epochs = 10
    device = "cuda"
    exps_dir = "exps"
    exp_name = "vgg19"
    save_name = "model_state.pt"

    model = "vgg19"
    freeze = False

    if not os.path.exists(exps_dir):
        os.mkdir(exps_dir)
    
    exp_path = os.path.join(exps_dir, exp_name)
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    # logger
    logger = create_logger(os.path.join(exp_path, exp_name))

    # data
    train_data = OurDataset("dataset/chest_xray/train")
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    val_data = OurDataset("dataset/chest_xray/val")
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # model
    if model == "alexnet":
        m = AlexNetCustom(n_classes)
    else:
        m = VGG19Custom(n_classes)

    if freeze:
        m.freezeFeatures()

    m.to(device)
    logger.info(f"Model: {m}")

    # optimizer and loss
    optimizer = optim.SGD(m.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()


    data_dict = {
        "weight": None,
        "best_weight": None
    }

    ## trainer
    t = 0
    best_acc=0
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        logger.info(f"New epoch: {epoch+1}")
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = m(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % do_val == 0:
                ## val
                m.eval()
                with torch.no_grad():
                    labels_val = []
                    outputs_val = []
                    for i, data in enumerate(valloader, 0):
                        inputs, labels = data[0].to(device), data[1].to(device)
                        output = m(inputs) # (b,2) -> softmax 
                        # pdb.set_trace()
                        output = F.softmax(output, dim=1).detach().cpu().numpy()[:,1] # output[:, 1]

                        for _i, _j in zip(np.array(data[1]), output):
                            labels_val.append(_i) 
                            outputs_val.append(0 if _j < 0.5 else 1) # (4,1)

                m.train()
                ## métrica
                acc = accuracy_score(y_true=labels_val, y_pred=outputs_val)
                logger.info(f"[Validation] accuracy: {acc}")
                writer.add_scalar("acc", acc, t+1)
                if acc > best_acc:
                    best_acc = acc
                    logger.info(f"[Validation] New best accuracy: {best_acc}")
                    logger.info(f"[Validation] Save model...")
                    data_dict["best_weight"] = m.state_dict()
                    torch.save(data_dict, os.path.join(exp_path, save_name))

            if i % print_args == 0:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #     (epoch + 1, i + 1, running_loss / print_args))
                logger.info(f"[Train loss] {epoch+1}-{t+1}: {loss.item()}")
                running_loss = 0.0
                writer.add_scalar("loss", loss.item(), t+1)
            
            t += 1

    
    data_dict["weight"] = m.state_dict()
    torch.save(data_dict, os.path.join(exp_path, save_name))