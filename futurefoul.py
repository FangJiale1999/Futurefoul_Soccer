import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import timeit
from PIL import Image
import numpy as np
import scipy.io
import torchvision.models.inception as inception
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActionModelPic(nn.Module):
    def __init__(self):
        super(ActionModelPic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2048, 512)
        self.gru = nn.GRU(512, hidden_size=256, num_layers=2, batch_first=True,dropout=0.2)#0.2


    def forward(self, x):        
        batch_size = x.size(0)
        seq_length = x.size(1)
        num_channels = x.size(2)
        height = x.size(3)
        width = x.size(4)

        x = x.view(batch_size * seq_length, num_channels, height, width)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = self.bn3(out)
        out = self.flatten(out)
        out = self.linear1(out)

        out = out.view(batch_size, seq_length, -1)
        out, _ = self.gru(out)
        out = out[:, -1, :]
        return out
    
    
class ActionModelBbox(nn.Module):
    def __init__(self):
        super(ActionModelBbox, self).__init__()
        self.gru = nn.GRU(10, 10, num_layers=3, batch_first=True, dropout=0.01)

    def forward(self, x):
        out, _ = self.gru(x)
        out=out[:, -1, :]
        return out

class ActionModelBboxPic(nn.Module):
    def __init__(self):
        super(ActionModelBboxPic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.2)
   
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()

    def forward(self, x):
        batch_size = x.size(0)
        seq_size = x.size(1)
        seq_length = x.size(2)
        num_channels = x.size(3)
        height = x.size(4)
        width = x.size(5)
        x = x.view(batch_size * seq_length * seq_size, num_channels, height, width)
        
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.bn1(out)
        out = self.dropout1(out)
               
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
             
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = self.bn3(out)
        out = self.flatten(out)
        out = out[:32, :]
        
        return out


class ActionModelPose(nn.Module):
    def __init__(self):
        super(ActionModelPose, self).__init__()
        self.gru = nn.GRU(255, 255, num_layers=3, batch_first=True, dropout=0.2)

    def forward(self, x):
        x = x.to(torch.float32)
        out, _ = self.gru(x)
        out=out[:, -1, :]
        return out


class ActionModel(nn.Module):
    def __init__(self):
        super(ActionModel, self).__init__()
        self.embedding_pic_dim=128
        self.embedding_bbox_dim=8
        self.embedding_bboxpic_dim=64
        self.embedding_pose_dim=8


        self.pic_model = ActionModelPic()
        self.bbox_model = ActionModelBbox()
        self.bboxpic_model = ActionModelBboxPic()
        self.pose_model = ActionModelPose()
        
        
        self.embedding_pic=nn.Linear(256,self.embedding_pic_dim*2)
        self.dropout3 = nn.Dropout(0.2)
        self.linear4 = nn.Linear(self.embedding_pic_dim*2, self.embedding_pic_dim)
        
        self.embedding_bbox=nn.Linear(10,self.embedding_bbox_dim)
        
        self.embedding_bboxpic=nn.Linear(512,self.embedding_bboxpic_dim)
        
        self.embedding_pose=nn.Linear(255,self.embedding_pose_dim*24)
        self.linear5 = nn.Linear(self.embedding_pose_dim*24, self.embedding_pose_dim)
        
        
        self.linear1=nn.Linear(self.embedding_pic_dim+self.embedding_bbox_dim+self.embedding_pose_dim+self.embedding_bboxpic_dim
                               ,self.embedding_pic_dim+self.embedding_bbox_dim+self.embedding_pose_dim+self.embedding_bboxpic_dim)
        self.linear2=nn.Linear(self.embedding_pic_dim+self.embedding_bbox_dim+self.embedding_pose_dim+self.embedding_bboxpic_dim,2)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pic_inputs, bbox_inputs,pose_inputs,bboxpic_inputs):
        pic_outputs = self.pic_model(pic_inputs)
        bbox_outputs = self.bbox_model(bbox_inputs)
        pose_outputs = self.pose_model(pose_inputs)
        bboxpic_outputs=self.bboxpic_model(bboxpic_inputs)#torch.Size([32, 2048])
        
        pic_outputs=self.embedding_pic(pic_outputs) # linear(256,256)
        pic_outputs=self.dropout3(pic_outputs)#dropout(0.2)
        pic_outputs=self.linear4(pic_outputs)#linear(256,128)
        
        bbox_outputs=self.embedding_bbox(bbox_outputs)#linear(10,2)
        bboxpic_outputs=self.embedding_bboxpic(bboxpic_outputs)#linear(2048,2)

        pose_outputs=self.embedding_pose(pose_outputs)#linear(255,48)
        pose_outputs=self.linear5(pose_outputs)#linear(48,2)
        

        bbox=torch.cat((bboxpic_outputs, bbox_outputs), dim=1)
        
        poseandbbox=torch.cat((pose_outputs, bbox), dim=1)
        
        combined_outputs = torch.cat((pic_outputs, poseandbbox), dim=1)
        
        out = self.dropout(combined_outputs) 
        out = F.relu(self.linear1(out))
        out = self.dropout(out) 
        out = F.relu(self.linear1(out))
        out = self.dropout(out) 
        out = F.relu(self.linear1(out))
        out = self.dropout(out) 
        out=self.linear1(out)
        out = self.softmax(self.linear2(out))
        return out    

class ActionDataset(Dataset):
    def __init__(self, root_dir, labels=[], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.length = len(os.listdir(self.root_dir))
        self.labels = labels + 1  

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        folder = idx + 1
        folder = format(folder, '05d')
        img_paths = [
            os.path.join(self.root_dir, folder, f'{i}.png')
            for i in range(1, 5)  
        ]
        image_list = []
        
        for img_path in img_paths:
            image = Image.open(img_path)

            if self.transform:
                image = self.transform(image)

            image_list.append(image)  
            
        images_tensor = torch.stack(image_list)  
        if len(self.labels) != 0:
            Label = self.labels[idx][0] - 1
            sample = {'images': images_tensor, 'img_paths': img_paths, 'Label': Label}
        else:
            sample = {'images': images_tensor, 'img_paths': img_paths}

        return sample
    
class ActionDataset1(Dataset):
    def __init__(self, root_dir, labels=[], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.length = len(os.listdir(self.root_dir))
        self.labels = labels + 1  

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        folder = format(idx + 1, '05d')
        txt_paths = [
            os.path.join(self.root_dir, folder, f'{i}.txt')
            for i in range(1, 5)  
        ]
        txt_data_list = []

        for txt_path in txt_paths:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                txt_data = [float(num) for line in lines for num in line.split(',')[2:]]
                foot_positions = [] 
                for i in range(0, len(txt_data), 4):
                    bbox = txt_data[i:i+4]  
                    bbox_center_x = bbox[0] + (bbox[2] / 2)
                    bbox_center_y = bbox[1] + (bbox[3] / 2)
                    foot_x = bbox_center_x - (bbox[2] / 2)
                    foot_y = bbox_center_y + (bbox[3] / 2)
                    position=[foot_x,foot_y]
                    foot_positions.append(foot_x)
                    foot_positions.append(foot_y)

                txt_data_list.append(foot_positions)

        txt_data_tensor = torch.tensor(txt_data_list)
        sample = {'txt': txt_data_tensor, 'txt_paths': txt_paths}

        if len(self.labels) != 0:
            Label = self.labels[idx][0] - 1
            sample['Label'] = Label

        return sample


class ActionDatasetBBox_pic(Dataset):

    def __init__(self, root_dir,pic_root, labels=[], transform=None):
        self.pic_root=pic_root
        self.root_dir = root_dir
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        self.length = len(os.listdir(self.root_dir))
        self.labels = labels + 1  

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        folder = format(idx + 1, '05d')
        txt_paths = [
            os.path.join(self.root_dir, folder, f'{i}.txt')
            for i in range(1, 5)  
        ]
        txt_data_list = []
        img_paths = [
            os.path.join(self.pic_root, folder, f'{i}.png')
            for i in range(1, 5)  
        ]
        image_list = []

        for txt_path,img_path in zip(txt_paths,img_paths):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                txt_data = [float(num) for line in lines for num in line.split(',')[2:]]
                image = Image.open(img_path)

                foot_positions = []
                bbox_imgs=[] 
                for i in range(0, len(txt_data), 4):
                    bbox = txt_data[i:i+4]  
                    x, y, w, h = map(int, bbox)  
                    left = x
                    upper = y
                    right = x + w
                    lower = y + h
                    bbox_image = image.crop((left, upper, right, lower))
                    bbox_image_tensor = self.transform(bbox_image)
                    bbox_imgs.append(bbox_image_tensor)
                    # save bboximg
                    # img_filename = os.path.basename(img_path)
                    # img_name, img_extension = os.path.splitext(img_filename)
                    # bbox_filename = f"{img_name}_{i}_{idx:02d}.jpg"  
                    # bbox_filepath = os.path.join(os.path.dirname(img_path), bbox_filename)
                    # bbox_image.save(bbox_filepath)         

                txt_data_list.append(torch.stack(bbox_imgs))

        txt_data_tensor = torch.stack(txt_data_list) 
        sample = {'txt': txt_data_tensor, 'txt_paths': txt_paths}
        if len(self.labels) != 0:
            Label = self.labels[idx % len(self.labels)][0] - 1
            sample['Label'] = Label

        return sample


class ActionDataset2(Dataset):
    """pose dataset."""
    def __init__(self, root_dir, labels=[], transform=None):
        self.root_dir = root_dir
        self.folders = sorted(os.listdir(self.root_dir))
        self.length = len(self.folders)
        self.labels = labels + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        folder = self.folders[idx]
        folder_path = os.path.join(self.root_dir, folder)
        
        txt_paths = [
            os.path.join(folder_path, f'{i}.txt')
            for i in range(1, 5)  
        ]
        max_size =  torch.tensor(0)
        txt_data_list = []
        for txt_path in txt_paths:
            with open(txt_path, 'r') as f:
                content = f.read() 
                data = eval(content)
                arr = np.array(data)
                arr_1d = arr.flatten()
                txt_data_list.append(torch.tensor(arr_1d))  
                max_size = max(max_size, arr_1d.size)

        target_size = torch.Size([4, 255])
        txt_data_list_padded = [
            F.pad(t, (0, target_size[1] - t.size(0)), value=0) for t in txt_data_list
        ] 
        txt_data = torch.stack(txt_data_list_padded, dim=0)
        sample = {'txt': txt_data, 'txt_paths': txt_paths}

        if len(self.labels) != 0:
            Label = self.labels[idx][0] - 1
            sample['Label'] = Label

        return sample



def pad_labels(labels, max_length):
    padded_labels = F.pad(labels, (0, max_length - len(labels)))
    return padded_labels

def predict_on_test_precision_recall(model, image_loader, bbox_loader, pose_loader,bboxpic_loader):
    num_correct = 0
    num_samples = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    num_after500=0
    num1=0
    model.eval() 
    results = open('results.csv', 'w')
    count = 0
    results.write('Id' + ',' + 'Class' + '\n')
    predict_list=[]
    for t, (image_data, bbox_data, pose_data,bboxpic_data) in enumerate(zip(image_loader, bbox_loader, pose_loader,bboxpic_loader)):
        pic_inputs, pic_labels = image_data['images'], image_data['Label']
        bbox_inputs, bbox_labels = bbox_data['txt'], bbox_data['Label']
        bboxpic_inputs, bboxpic_labels = bboxpic_data['txt'], bboxpic_data['Label']
        pose_inputs, pose_labels = pose_data['txt'], pose_data['Label']
        pic_inputs = pic_inputs.to(device)
        pic_labels = pic_labels.to(device)
        bbox_inputs = bbox_inputs.to(device)
        bbox_labels = bbox_labels.to(device)
        pose_inputs = pose_inputs.to(device)
        pose_labels = pose_labels.to(device)
        bboxpic_inputs=bboxpic_inputs.to(device)
        bboxpic_labels=bboxpic_labels.to(device)
        max_length = max(pic_labels.size(0), bbox_labels.size(0))
        pic_labels = pad_labels(pic_labels, max_length)
        bbox_labels = pad_labels(bbox_labels, max_length)

        with torch.no_grad():

            outputs = model(pic_inputs,bbox_inputs,pose_inputs,bboxpic_inputs)
            _, preds = outputs.data.max(1)
            num_correct += (preds == pic_labels).sum().item()
            true_positives += ((preds == 1) & (pic_labels == 1)).sum().item()
            false_positives += ((preds == 1) & (pic_labels == 0)).sum().item()
            false_negatives += ((preds == 0) & (pic_labels == 1)).sum().item()
            
            num_samples += pic_labels.size(0)
            predict_list.append(preds)
            for i in range(len(preds)):
                results.write(str(count) + ',' + str(preds[i]) + '\n')
                count += 1
    flattened_list = [tensor.flatten().tolist() for tensor in predict_list]
    df = pd.DataFrame({'column_name': flattened_list})
    df.to_csv("/output1.csv",index=False)

    results.close()


    test_accuracy = num_correct / num_samples
    if true_positives + false_positives!=0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        print('Precision: {:.2f}'.format(precision))
        print('Recall: {:.2f}'.format(recall))
    else :
        print("true_positives + false_positives=0")
    return count, test_accuracy

#1->foul 0->nonfoul
label_mat = scipy.io.loadmat('./train_val.mat')
label_train = label_mat['trLb']
print('train len:', len(label_train))
label_val = label_mat['valLb']
print('val len:', len(label_val))

image_dataset_train = ActionDataset(
    root_dir='./train/pic',
    labels=label_train, transform=T.Compose([T.Resize((64, 64)), T.ToTensor()]))

image_dataloader_train = DataLoader(image_dataset_train, batch_size=32, shuffle=True, num_workers=20)

image_dataset_val = ActionDataset(
    root_dir='./val/pic',
    labels=label_val, transform=T.Compose([T.Resize((64, 64)), T.ToTensor()]))

image_dataloader_val = DataLoader(image_dataset_val, batch_size=32, shuffle=False, num_workers=20)

bbox_dataset_train = ActionDataset1(root_dir='./train/bbox', labels=label_train)
bbox_dataloader_train = DataLoader(bbox_dataset_train, batch_size=32, shuffle=True, num_workers=20)

bbox_dataset_val = ActionDataset1(root_dir='./val/bbox', labels=label_val)
bbox_dataloader_val = DataLoader(bbox_dataset_val, batch_size=32, shuffle=False, num_workers=20)

pose_dataset_train = ActionDataset2(root_dir='./train/pose', labels=label_train)
pose_dataloader_train = DataLoader(pose_dataset_train, batch_size=32, shuffle=True, num_workers=20)

pose_dataset_val = ActionDataset2(root_dir='./val/pose', labels=label_val)
pose_dataloader_val = DataLoader(pose_dataset_val, batch_size=32, shuffle=False, num_workers=20)

bboxpic_dataset_train = ActionDatasetBBox_pic(root_dir='./train/bbox', 
                                    pic_root='./train/pic',
                                    labels=label_train)
bboxpic_dataloader_train = DataLoader(bboxpic_dataset_train, batch_size=32, shuffle=True, num_workers=20)

bboxpic_dataset_val = ActionDatasetBBox_pic(root_dir='./val/bbox', 
                                  pic_root='./val/pic',
                                  labels=label_val)
bboxpic_dataloader_val = DataLoader(bboxpic_dataset_val, batch_size=32, shuffle=False, num_workers=20)

image_dataset_test=ActionDataset(root_dir='./test/pic',
                                 labels=label_val, transform=T.Compose([T.Resize((64, 64)), T.ToTensor()]))

image_dataloader_test = DataLoader(image_dataset_test, batch_size=32,shuffle=False, num_workers=20)

bbox_dataset_test=ActionDataset1(root_dir='./test/bbox',
                                 labels=label_val)

bbox_dataloader_test = DataLoader(bbox_dataset_test, batch_size=32,
                        shuffle=False, num_workers=20)

pose_dataset_test=ActionDataset2(root_dir='./test/pose',
                                 labels=label_val)

pose_dataloader_test = DataLoader(pose_dataset_test, batch_size=32,
                        shuffle=False, num_workers=20)

bboxpic_dataset_test = ActionDatasetBBox_pic(root_dir='./test/bbox', 
                                  pic_root='./test/pic',
                                  labels=label_val)
bboxpic_dataloader_test = DataLoader(bboxpic_dataset_val, batch_size=32, shuffle=False, num_workers=20)




model = ActionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

num_epochs = 200
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
early_stop=float('inf') 
early_stop_count=0


for epoch in range(num_epochs):
    running_train_loss = 0.0
    running_train_correct = 0
    running_train_total = 0
    
    running_val_loss = 0.0
    running_val_correct = 0
    running_val_total = 0
    

    model.train()
    for i, data in enumerate(zip(image_dataloader_train, bbox_dataloader_train, pose_dataloader_train,bboxpic_dataloader_train)):

        image_data, bbox_data, pose_data,bboxpic_data = data

        pic_inputs, pic_labels = image_data['images'], image_data['Label']
        bbox_inputs, bbox_labels = bbox_data['txt'], bbox_data['Label']
        bboxpic_inputs, bboxpic_labels = bboxpic_data['txt'], bboxpic_data['Label']
        pose_inputs, pose_labels = pose_data['txt'], pose_data['Label']
        
        pic_inputs = pic_inputs.to(device)
        pic_labels = pic_labels.to(device)

        bbox_inputs = bbox_inputs.to(device)
        bbox_labels = bbox_labels.to(device)

        pose_inputs = pose_inputs.to(device)
        pose_labels = pose_labels.to(device)
        
        bboxpic_inputs=bboxpic_inputs.to(device)
        bboxpic_labels=bboxpic_labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(pic_inputs,bbox_inputs,pose_inputs,bboxpic_inputs)

        loss = criterion(outputs, pic_labels)
        loss.backward()
        optimizer.step()


        running_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_train_total += pic_labels.size(0)
        running_train_correct += (predicted == pic_labels).sum().item()

    train_epoch_loss = running_train_loss / len(image_dataloader_train)
    train_epoch_accuracy = running_train_correct / running_train_total

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(zip(image_dataloader_val, bbox_dataloader_val, pose_dataloader_val,bboxpic_dataloader_train)):
            image_data, bbox_data, pose_data,bboxpic_data = data
            pic_inputs, pic_labels = image_data['images'], image_data['Label']
            bbox_inputs, bbox_labels = bbox_data['txt'], bbox_data['Label']
            bboxpic_inputs, bboxpic_labels = bboxpic_data['txt'], bboxpic_data['Label']
            pose_inputs, pose_labels = pose_data['txt'], pose_data['Label']
            
            pic_inputs = pic_inputs.to(device)
            pic_labels = pic_labels.to(device)

            bbox_inputs = bbox_inputs.to(device)
            bbox_labels = bbox_labels.to(device)

            pose_inputs = pose_inputs.to(device)
            pose_labels = pose_labels.to(device)
            
            bboxpic_inputs=bboxpic_inputs.to(device)
            bboxpic_labels=bboxpic_labels.to(device)
            outputs = model(pic_inputs,bbox_inputs,pose_inputs,bboxpic_inputs)
            loss = criterion(outputs, pic_labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_val_total += pic_labels.size(0)
            running_val_correct += (predicted == pic_labels).sum().item()

    val_epoch_loss = running_val_loss / len(image_dataloader_val)
    
    if val_epoch_loss< early_stop:
        early_stop=val_epoch_loss
        early_stop_count = 0 
        optimal_model_param=model.state_dict()
    else:
        early_stop_count+=1
        if early_stop_count>=16: 
            print(f"Early stopped at epoch {epoch-15}")
            break

    val_epoch_accuracy = running_val_correct / running_val_total
    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_accuracies.append(val_epoch_accuracy)

    print('Epoch [{}/{}], Training Loss: {:.4f}, Training Accuracy: {:.2f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, train_epoch_loss, train_epoch_accuracy * 100, val_epoch_loss, val_epoch_accuracy * 100))
    
count, test_accuracy = predict_on_test_precision_recall(model, image_dataloader_test, bbox_dataloader_test, pose_dataloader_test,bboxpic_dataloader_test)
print('Test Accuracy: {:.2f}%'.format(test_accuracy * 100))
print(f'futureFoul.py,num_epochs:{num_epochs}')    