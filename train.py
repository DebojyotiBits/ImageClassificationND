#importing necessary libraries



import matplotlib.pyplot as plt

import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json
from sklearn import metrics as skmetrics
import numpy

import numpy as np


import torchvision.transforms as transform
import torchvision
from torchvision import datasets, transforms, models
from torchvision import models

import time

from tqdm import tqdm
import logging
import numpy as np
from datetime import datetime

# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 
torch.cuda.empty_cache()





time_str = str(datetime.now().strftime("%Y%m%d-%H%M"))
# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of training script")

parser.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', help = 'Vgg16', type = str)
parser.add_argument ('--learning_rate', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 2048', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--gpu',  dest='gpu', action='store_true',help = "Option to use GPU")
print("Arguments loaded")
#setting values data loading
args = parser.parse_args ()
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#defining device: either cuda or cpu
if args.gpu:
    print("Device GPU")
    device = 'cuda'
else:
    device = 'cpu'

#data loading
if data_dir: #making sure we do have value for data_dir
    # Define your transforms for the training, validation, and testing sets
    mean = (0.4124234616756439, 0.3674212694168091, 0.2578217089176178)
    std = (0.3268945515155792, 0.29282665252685547, 0.29053378105163574)
    transformer = transform.Compose([
                           transform.Resize((224, 224)),
                           transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                           transform.RandomRotation(5),
                           transform.RandomAffine(degrees=11, translate=(0.1,0.1), scale=(0.8,0.8)),
                           transform.ToTensor(),
                           transform.Normalize(mean, std)])
                                       
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=transformer)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=transformer)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=transformer) # cost
  
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    image_datasets = [train_data, valid_data, test_data]
    dataloaders = [trainloader, validloader, testloader]
    
#Load the optimizer 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()




print("Loaders defined")
#mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
no_output_categories = len(cat_to_name)
print('no_output_categories',no_output_categories)

class custom_metric:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        # initialize a metric dictionary
        self.metric_dict = {metric_name: [0] for metric_name in self.metric_names}

    def Step(self, labels, preds):
        for metric in self.metric_names:
            # get the metric function
            do_metric = getattr(
                skmetrics, metric, "The metric {} is not implemented".format(metric)
            )
         
            try:
                self.metric_dict[metric].append(
                    do_metric(labels, preds, average="macro")
                )
            except:
                self.metric_dict[metric].append(do_metric(labels, preds))

    def Epoch(self):
        # calculate metrics for an entire epoch
        avg = [sum(metric) / (len(metric) - 1) for metric in self.metric_dict.values()]
        metric_as_dict = dict(zip(self.metric_names, avg))
        return metric_as_dict

    def last_step_metric(self):
        # return metrics of last steps
        values = [self.metric_dict[metric][-1] for metric in self.metric_names]
        metric_as_dict = dict(zip(self.metric_names, values))
        return metric_as_dict



def load_model (hidden_units):
    torch.cuda.empty_cache()
    model = models.vgg16_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, no_output_categories)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    model = model.to(device)
    return model

  
    

#Funvtion to train per epoch using the model
def train_epoch(model,train_loader,test_loader,device,optimizer,criterion, train_metrics, val_metrics):

    # training-the-model
    train_loss = 0
    valid_loss = 0
    all_labels = []
    all_preds = []
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.type(torch.FloatTensor).to(device)
        target = target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        output = model(data)
        preds = torch.argmax(output, axis=1).cpu().detach().numpy()
        labels = target.cpu().numpy()
        
        # calculate-batch-loss
        loss = criterion(output.type(torch.FloatTensor), target.type(torch.LongTensor))
        loss.backward()
        optimizer.step()
        
        # updatetraining-loss
        train_loss += loss.item() * data.size(0)
        # calculate training metrics
        all_labels.extend(labels)
        all_preds.extend(preds)
    
    train_metrics.Step(all_labels, all_preds)

    # validate-model
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.type(torch.FloatTensor).to(device)
            target = target.to(device)
            output = model(data)
            preds = torch.argmax(output, axis=1).tolist()
            labels = target.tolist()
            all_labels.extend(labels)
            all_preds.extend(preds)
            loss = criterion(output, target)

            # update-average-validation-loss
            valid_loss += loss.item() * data.size(0)

    val_metrics.Step(all_labels, all_preds)
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(test_loader.sampler)

    return (train_loss, valid_loss, train_metrics.last_step_metric(), val_metrics.last_step_metric())








#Train Model

if args.learning_rate: #if learning rate was provided
    hidden_units = args.hidden_units
else:
    hidden_units = 4096
model=load_model(hidden_units)

if args.learning_rate: #if learning rate was provided
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.learning_rate)
else:
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.5)  

if args.epochs:
    num_epoch = args.epochs
else:
    num_epoch=20
train_metrics = custom_metric(["accuracy_score", "f1_score"])
val_metrics = custom_metric(["accuracy_score", "f1_score"])    
    
    
best_val_acc = 0

print("Begin training.........")

for i in tqdm(range(0, num_epoch)):
    loss, val_loss, train_result, val_result = train_epoch(model, trainloader, validloader, device, optimizer,
        criterion, train_metrics,val_metrics)

    scheduler.step(val_loss)
    print("Epoch {} / {} \n Training loss: {} - Other training metrics: ".format(i + 1, num_epoch, loss))
    print(train_result)
    print(" \n Validation loss : {} - Other validation metrics:".format(val_loss))
    print(val_result)
    print("\n")
    # saving epoch with best validation accuracy
    if best_val_acc < float(val_result["accuracy_score"]):
        print("Validation acc. = "+  str(val_result["accuracy_score"])+ "---- Saving best epoch")
        best_val_acc = val_result["accuracy_score"]
        torch.save(model.state_dict(),"./" +  "best.pt")
    else:
        print("Validation accuracy= "+ str(val_result["accuracy_score"]))
        continue
        
        
        
          
        
#saving trained Model
model.to ('cpu') #no need to use cuda for saving/loading model.
# Save the checkpoint
model.class_to_idx = train_data.class_to_idx #saving mapping between predicted class and class name,

checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': "vgg16",
              'mapping':    model.class_to_idx
             }
#saving trained model for future use
if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')