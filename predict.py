#importing necessary libraries
import matplotlib.pyplot as plt
import torch
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
import torchvision.transforms as transform
import torchvision
from torchvision import datasets, transforms, models
from torchvision import models



# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of prediction script")

parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
parser.add_argument ('--gpu',  dest='gpu', action='store_true',help = "Option to use GPU")



# a function that loads a checkpoint and rebuilds the model
def loading_model (file_path):
    torch.cuda.empty_cache()
    model = models.vgg16_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    if device=="gpu":
        map_location=lambda device, loc: device.cuda()
    else:
        map_location='cpu'     
    checkpoint = torch.load(f=file_path,map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)
    return model,checkpoint['mapping']


def process_image(image):
    processed_image = Image.open(image).convert('RGB')
    # Resize the image
    processed_image.thumbnail(size=(256,256)) 
    width, height = processed_image.size

    new_width,new_height = 224,224 
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    processed_image = processed_image.crop((left, top, right, bottom))

    transf_tens = transforms.ToTensor()
    transf_norm = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    tensor = transf_norm(transf_tens(processed_image))
    np_processed_image = np.array(tensor)
    return np_processed_image


def class_to_label(file,classes):
    with open(file, 'r') as f:
        class_mapping =  json.load(f)     
    labels = []
    for c in classes:
        labels.append(class_mapping[c])
    return labels


#defining prediction function
def predict(image_path, model,idx_mapping, topk, device):
    # defines preprocessed image
    pre_processed_image = torch.from_numpy(process_image(image_path))
    pre_processed_image = torch.unsqueeze(pre_processed_image,0).to(device).float()
    model.to(device)
    model.eval()
    
    log_ps = model.forward(pre_processed_image)
    ps = torch.exp(log_ps)
    top_ps,top_idx = ps.topk(topk,dim=1)
    list_ps = top_ps.tolist()[0]
    list_idx = top_idx.tolist()[0]
    classes = []
    model.train()
    for x in list_idx:
        classes.append(idx_mapping[x])
    return list_ps, classes

def print_predictions(probabilities, classes,image,category_names=None):
    # prints out the image
    print(image)
    
    if category_names:
        labels = class_to_label(args.category_names,classes)
        for i,(ps,ls,cs) in enumerate(zip(probabilities,labels,classes),1):
            print(f'{i}) {ps*100:.2f}% {ls.title()} - Class No. {cs}')
    else:
        for i,(ps,cs) in enumerate(zip(probabilities,classes),1):
            print(f'{i}) {ps*100:.2f}% Class No. {cs} ')
    print('') 



#setting values data loading
args = parser.parse_args ()
file_path = args.image_dir

#defining device: either cuda or cpu
if args.gpu :
    device = 'cuda'
else:
    device = 'cpu'

#loading JSON file if provided, else load default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#loading model from checkpoint provided
model,class_to_idx = loading_model (args.load_dir)

#defining number of classes to be predicted. Default = 1
if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1
    
    
    

    
idx_mapping = dict(map(reversed, class_to_idx.items()))    
probabilities,classes = predict(args.image_dir,model,idx_mapping,nm_cl,device)
print_predictions(probabilities,classes,args.image_dir.split('/')[-1],args.category_names)