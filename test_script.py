import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

# This function will crop image and add small padding which is needed to perform the preprocessing for inference of the model 
def crop_img(img, bbox):
    mdim = max(wbox, hbox) 
    pad = int(mdim * 0.15 * (1 + np.random.rand()))
    
    # print("pad value: {}".format(pad))

    x_start, x_end = bbox[0] - pad, bbox[2] + pad
    y_start, y_end = bbox[1] - pad, bbox[3] + pad

    # Random cropping in some range 
    cropped_img = image_out[max(y_start,0):min(image_out.shape[0],y_end),  
                            max(x_start,0):min(image_out.shape[1],x_end), :]
    
    # converting the image to pil for applieng the transforms correctly 
    cropped_img = Image.fromarray(crop_cropped_imgimg)
    return cropped_img 

def preprocess_imgs(batch_imgs, batch_bboxes):
    img_transforms = v2.Compose([v2.Resize((224,224)),
                                v2.ToTensor(),
                                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    img_processed = []

    # Iterating over the images and applieng the transformation and cropping before passing it to the model 
    for id in range(0, len(batch_imgs)):
        img = batch_imgs[id]
        bbox = batch_boxes[id]
        img_crop = crop_img(img, bbox)

        img_tformed = img_transforms(img_crop)
        img_processed.append(img_tformed)

    img_batch = torch.stack([img_processed])
    print("processed image batch shape: {}".format(img_batch.shape))
    return img_batch 

# this function will the test the model on the given list of images, by applyieng the transform
def test_model(model, batch_imgs, batch_bboxes, gt_poses): 
    # Processing the images by the given transformations to pass to the learned model 
    batch_imgs = preprocess_imgs(batch_imgs, batch_bboxes)
    batch_imgs = batch_imgs.to(device)

    pred = model(batch_imgs)