import tensorflow as tf
import torch
import torchvision.models as models
from tensorflow.keras.applications import VGG16, ResNet50

# load a pre-trained model ResNet50 model
model_tf = ResNet50(weights='imagenet')
model_torch = models.resnet50(pretrained=True)

print(model_torch)

# display the models architecture
model_tf.summary()


# freeze the model parameters
for param in model_torch.parameters():
    param.requires_grad = False

# modify the final layer for a new task
num_features = model_torch.fc.in_features
model_torch.fc = torch.nn.Linear(num_features, 10)  # assuming 10
print('\n\nModified Torch Model Architecture:\n', model_torch)





# Access specifik layers
for i, layer in enumerate(model_tf.layers):
    print(f'\n\nLayer {i}: {layer.name}, Trainable: {layer.trainable}')


for name, param in model_torch.named_parameters():
    print(f'\n\nLayer: {name}, Required Grad: {param.requires_grad}')