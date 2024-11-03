import torch
import torch.nn as nn 
import numpy as np
import antialiased_cnns 

# This class will create a pose predictor on top of a resnet model along with a few set of linear layers 
class pose_model(nn.Module):
    def __init__(self):
        super(pose_model, self).__init__()

        model = antialiased_cnns.resnet18(pretrained=True)
        layers = list(model.children())[:-1]
        
        # Defining the feature extractors to extract features from a predefined model 
        self.feature_extractor = nn.Sequential(*layers) 

        # Requires grad is on for the model 
        # for param in self.feature_extractor.parameters():
        #     print(param.requires_grad)

        # A small projector head to predict the pose from resnet features 
        self.projector = nn.Sequential(*[nn.Linear(512, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128,1)])

    def forward(self, x):
        y = self.feature_extractor(x)
        y = y.view(y.shape[0],-1) # flattening the vector to pass it through the model 
        # print("pred y shape: {}".format(y.shape))
        y = self.projector(y) 
        # exit()
        # Multiplieng 2*pi with the sigmoid value which is in the range of (0,1)
        y = nn.Sigmoid()(y) * (2 * torch.pi)

        return y 

if __name__ == "__main__":
    model = pose_model()
    inputs = torch.randn(8,3,224,224)
    outputs = model(inputs)

    print("model inputs shape: {}".format(inputs.shape))
    print("model outputs shape: {}".format(outputs.shape))
