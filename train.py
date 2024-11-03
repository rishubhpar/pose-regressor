import cv2
import numpy as np
import torch
import torch.nn as nn 
from dataset import pose_data
from pose_regressor import pose_model
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset 
import matplotlib.pyplot as plt 
from bounding_box_vis import draw_azimuth
from torchvision.transforms import v2 

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

# Set a fixed random seed for reproducibility
torch.manual_seed(42) 

# To capture the circular nature of the angles and computing the loss between angles by projecting 
# them as a point in the unit circle and capturing the angle between the two vectors.
# hopefully, this will fix the sudden jumps in the validation losses  
def angular_loss(angle1, angle2):
    angle1 = angle1 % (2 * torch.pi)
    angle2 = angle2 % (2 * torch.pi)

    angle1_vector = torch.stack([torch.cos(angle1), torch.sin(angle1)], dim=-1)  # Shape: (N, 2)
    angle2_vector = torch.stack([torch.cos(angle2), torch.sin(angle2)], dim=-1)  # Shape: (N, 2)
    
    # Calculate the Euclidean distance
    loss = torch.norm(angle1_vector - angle2_vector, dim=1)  # Shape: (N,)
    
    return loss.mean()  # Return the mean to get a scalar loss 


# Training the model for one single epoch with the dataset and the denoising task 
def train_one_epoch(model, train_dl, optimizer, ep):
    losses = []

    for it,batch in enumerate(train_dl):
        img, azimuth = batch
        img, azimuth = img.to(device), azimuth.to(device)
        # print("X shape: {}, azim shape: {}".format(img.shape, azimuth.shape))

        # forward pass of the model with the input image 
        azimuth_pred = model(img)

        # cleaning up the gradients from the optimizer 
        optimizer.zero_grad()

        # Computing the loss between the predicted and original angle 
        # criterion = torch.nn.MSELoss()
        # print("azimuth pred: {}, azimuth: {}".format(azimuth_pred.squeeze().shape, azimuth.shape))
        # loss = criterion(azimuth_pred.squeeze(), azimuth) # converting the azimuth preds into a single vector
        
        # Testing teh angular loss for training between angles 
        loss = angular_loss(azimuth_pred.squeeze(), azimuth)
        loss.backward() # computing the gradients 
        optimizer.step() # Updating the values for parameters 

        losses.append(loss.item())
        
        if (it % 20 == 0):
            print("[{}] Epoch, [{}] iter, loss val: {:.4f}".format(ep, it, loss.item()))

    mean_loss = torch.tensor(losses).mean(axis=0)
    return mean_loss, losses


def validate_model(model, test_dl, ep):
    losses = []
    
    for it, batch in enumerate(test_dl):
        img, azimuth = batch
        img, azimuth = img.to(device), azimuth.to(device)

        # forward pass of the model with the input image 
        azimuth_pred = model(img)

        # Computing the loss between the predicted and original angle 
        #criterion = torch.nn.MSELoss()
        #loss = criterion(azimuth_pred.squeeze(), azimuth) # converting the azimuth preds into a single vector
        
        # computing the angular loss for more stability and to measure cyclic nature of angles 
        loss = angular_loss(azimuth_pred.squeeze(), azimuth)        
        losses.append(loss.item())

    mean_loss = torch.Tensor(losses).mean(axis=0)
    return mean_loss, losses

# Testing the models for a corresponding model path for inference 
def test_model(model, test_dl):    
    ckpt_path = './logs/ckpts/pose_reg_8.pth'
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    print("loaded model weights from: {}".format(ckpt_path)) 

    # Iterating over the batch to perform the predictions for testing
    for it, batch in enumerate(test_dl):
        img, azimuth = batch
        img, azimuth = img.to(device), azimuth.to(device)

        # forward pass of the model with the input image 
        azimuth_pred = model(img)
        # now we have to save the image and its corresponding azimuthal and predicted azimuthal 

        save_predictions(img, azimuth, azimuth_pred, it)


# This function will save the predictions along with the input image as a visualization 
def save_predictions(imgs, azimuths, azimuths_pred, it):
    imgs = list(imgs.unbind(0)) # Extract the images from the preds 
    imgs_save_paths = './logs/preds/' + str(it)

    # Define the normalization parameters
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Define the unnormalize transform
    unnormalize = v2.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    for id in range(0, len(imgs)):
        img = imgs[id]
        img = unnormalize(img) # bringing it back to the range of (0,1) after doung unnormalization 
        # print("img processed shape:{}".format(img.shape))
        # print("img min max: {}, {}".format(img.min(), img.max()))

        # getting the angles  to be displayed 
        gt_azimuth = azimuths[id]
        pred_azimuth = azimuths_pred[id]

        # print("gt azimuth: {}, pred azimuth: {}".format(gt_azimuth, pred_azimuth[0])) 

        # drawing the images for given orientation 
        img_gt_azimuth = draw_azimuth(img.shape[1], gt_azimuth.detach().cpu().numpy(), (180,190,255))  
        img_pred_azimuth = draw_azimuth(img.shape[1], pred_azimuth[0].detach().cpu().numpy(), (254,190,180)) 

        # permuting the tensor so that it looks like an image 
        img = img.detach().cpu().permute(1,2,0) 
        img = img.numpy()*255.0
        img = img.astype(np.uint8)

        # print("img shapes, img: {}, img_draw1: {}, img_draw2: {}".format(img.shape, img_gt_azimuth.shape, img_pred_azimuth.shape))

        combined_img = np.hstack([img, img_gt_azimuth, img_pred_azimuth]) 

        img_path = imgs_save_paths + '_' + str(id) + '.png'
        cv2.imwrite(img_path, combined_img) 


# Training model for the fixed number of epochs along with validation after each epochs with saving images 
def train():
    n_epochs = 100 # 10 # 1000
    batch_size = 128
    lr = 0.00005
    save_every = 1 # Saving checkpoint every 5 epochs 
    data_path = './data/pose_regressor_gt' # './data/training_data_2410/'

    # initializing model and datasets 
    model = pose_model().to(device)
    dataset = pose_data(data_path)

    # Creating the train test split for our custom pose dataset 
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)    

    # Use Subset to create train and test datasets from indices
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    print("train dataset shape: {}".format(len(train_dataset)))
    print("test dataset shape: {}".format(len(test_dataset)))

    # Creating a dataloader for the train and test subsets of data 
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # optimizer 
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    epoch_train_losses, epoch_val_losses = [], []

    for ep in range(n_epochs):
        epoch_loss, _ = train_one_epoch(model, train_dl, optimizer, ep)
        epoch_train_losses.append(epoch_loss)
        print("[{}/{}] Epoch Train Loss: {:.4f}".format(ep, n_epochs, epoch_loss.item()))
        val_loss, _ = validate_model(model, test_dl, ep)
        epoch_val_losses.append(val_loss) 
        print("[{}/{}] Epoch Test Loss: {:.4f}".format(ep, n_epochs, val_loss.item())) 

        if (ep%save_every == 0):
            ckpt_path = './logs/ckpts/large_data_pose_reg_' + str(ep) + '.pth'
            
            torch.save(model.state_dict(), ckpt_path)
            print("saving checkpoint at epoch: {}".format(ep))    

            x_axis = range(1, len(epoch_train_losses) + 1)
            plt.plot(x_axis, epoch_train_losses, label='Training Loss')
            plt.plot(x_axis, epoch_val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.legend()
            plt.savefig('./model_train.png')
            plt.clf()

    # Testig the model once the training is done and saving the outputs 
    test_model(model, test_dl)

if __name__ == "__main__":
    train()
