import tensorflow as tf
import os
from mpmath.identification import transforms
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# load pre-trained
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# freeze the base model
base_model.trainable = False


# add custom classification head
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)

output = Dense(5, activation='softmax')(x)

# create the final model
model_tf = Model(inputs=base_model.input, outputs=output)
model_tf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# data preparation
datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2
)

train_data = datagen.flow_from_directory(
    r'D:\PYTHON\Pycharm\Udemy - Full-Stack AI Engineer 2026 ML, Deep Learning, GenerativeAI\013_transfer learning and fine-tuning',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    r'D:\PYTHON\Pycharm\Udemy - Full-Stack AI Engineer 2026 ML, Deep Learning, GenerativeAI\013_transfer learning and fine-tuning',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# train the model
history = model_tf.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data),
)

for layer in base_model.layers[-5:]:
    layer.trainable = True

model_tf.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, loss='categorical_crossentropy', metrics=['accuracy']),
)

val_loss, val_accuracy = model_tf.evaluate(val_data)












import torch
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transformers
import torch.optim as optim


model_torch = models.resnet50(pretrained=True)
for param in model_torch.parameters():
    param.requires_grad = False

# replace the last layer for a new classification task
num_features = model_torch.fc.in_features
model_torch.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 5),
    nn.Softmax(dim=1)
)

print('\n\n')
print(model_torch)


# data preparation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

train_data = datasets.ImageFolder(
    'PATH_TO_FOLDER_PATH',
    transform=transform
)
val_data = dataset.ImageFolder(
    'PATH_TO_FOLDER_VAL',
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)


# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_torch.fc.parameters(), lr=0.002)


# training loop
for epoch in range(10):
    model_torch.fc.train()

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_torch.fc(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


    print(f"Epochs: {epoch+1}, Loss: {loss.item()}")


for name, param in model_torch.fc.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model_torch.fc(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:4f}%')