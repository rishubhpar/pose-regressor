import numpy as np
import torch
import os
import cv2
import pickle 
import torch.nn as nn
from PIL import Image 
from torchvision.transforms import v2

# Class for pose dataset for predicting pose from monocular images 
class pose_data:
    def __init__(self, path):
        self.data_path_render = path 
        self.data_path_augms = path + '_canny_controlnet' 

        # Listing all the object names and there copies 
        self.data_list = [cn for cn in os.listdir(self.data_path_render)]
        print("self data list: {}, items - {}".format(len(self.data_list), self.data_list))

        self.data_names = []
        for id in range(0, len(self.data_list)):
            obj_name = self.data_list[id]
            for file_ in os.listdir(os.path.join(self.data_path_augms,obj_name)):
                self.data_names.append(os.path.join(obj_name,file_)) 
        
        self.data_len = len(self.data_names) # For all the controlnet augmentations counting the number of images 

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.data_len)
        
        # augm image path remains the same
        augm_img_path = os.path.join(self.data_path_augms, self.data_names[idx]) 
        # print("the object properties: {}".format(augm_img_path))

        # loading the blender render image and bounding box 
        img_prefix = self.data_names[idx].split('____')[:-1] # getting the image name prefix 
        img_prefix = '_'.join(img_prefix)
        # print("img prefix: {}".format(img_prefix)) 

        # getting the angle from the last element 
        azimuth = img_prefix.split('_')[-1]
        azimuth = torch.tensor(float(azimuth), dtype=torch.float)
        # print("img azimuth angle: {}".format(azimuth))

        blender_img_path = os.path.join(self.data_path_render, img_prefix + '__.jpg')
        bbox_path = os.path.join(self.data_path_render, img_prefix + '__.pkl') 
        # print("blender img path: {}".format(blender_img_path))
        # print("bbox path: {}".format(bbox_path)) 

        with open(bbox_path, "rb") as file:
            data = pickle.load(file) 
        bbox = data['obj1']['bbox']

        # print("loaded box: {}".format(bbox))
        hbox, wbox = bbox[2]-bbox[0], bbox[3]-bbox[1]

        # Randomly picking blender image or augmented image 
        if (np.random.randint(0,10) < 1):
            image_out = cv2.imread(blender_img_path)
        else:
            image_out = cv2.imread(augm_img_path)

        # Performing data transformations 
        # Cropping with a random padding, need not be tight bounding box 
        mdim = max(wbox, hbox) 
        pad = int(mdim * 0.15 * (1 + np.random.rand()))
        
        # print("pad value: {}".format(pad))

        x_start, x_end = bbox[0] - pad, bbox[2] + pad
        y_start, y_end = bbox[1] - pad, bbox[3] + pad

        # Random cropping in some range 
        crop_img_raw = image_out[max(y_start,0):min(image_out.shape[0],y_end),  
                             max(x_start,0):min(image_out.shape[1],x_end), :]
        
        # for debugging  -------------------------------------------------- 

        # Saving the cropped image to just verify the framework 
        # imgsavepath = './debug/' + str(idx) + '.png'
        # print("saving image: {}".format(imgsavepath))
        # cv2.imwrite(imgsavepath, crop_img_raw)

        # converting the image to pil for applieng the transforms correctly 
        crop_img = Image.fromarray(crop_img_raw)

        img_transforms = v2.Compose([v2.Resize((224,224)),
                                     v2.ToTensor(),
                                     v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        crop_img_transform = img_transforms(crop_img) 
        # print("crop image transform {}".format(crop_img_transform.shape))

        return crop_img_transform, azimuth
    
    def __len__(self):
        return self.data_len
    

if __name__ == "__main__":
    path_data = './data/pose_regressor_gt'
    dataset = pose_data(path_data)

    for i, data in enumerate(dataset,0):
        crop_img, azimuth = data
        print("cropped image shape: {}, azimuth shape: {}".format(crop_img.shape, azimuth))
        # exit()

        if (i > 10):
            break


