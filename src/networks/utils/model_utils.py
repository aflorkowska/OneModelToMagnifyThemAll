import torch.nn as nn
from torchinfo import summary

def model_summary_with_pixel_size(model, input_size, pixel_size):
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.pixel_size = pixel_size
            
        def forward(self, x):
            return self.model(x, self.pixel_size)

    wrapped_model = ModelWrapper(model)
    summary(wrapped_model, input_size=input_size)
