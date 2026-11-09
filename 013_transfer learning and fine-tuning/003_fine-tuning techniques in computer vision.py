# import tensorflow as tf
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
# for layer in base_model.layers:
#     layer.trainable = False
#
# # add classification head
# x = GlobalAveragePooling2D()(base_model.output)
# output = Dense(5, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=output)
#
#
# # define data augmentation
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2,
# )
#
# train_data = datagen.flow_from_directory(
#     r"image_TL",
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training',
# )
#
# val_data = datagen.flow_from_directory(
#     "image_TL",
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation',
# )
#
# # compile the model
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy']
# )
#
# # train the model
# history = model.fit(
#     train_data, validation_data=val_data,
#     epochs=11, steps_per_epoch=len(train_data),
#     validation_steps=len(val_data),
# )
#
# print('\n\n\n')




import torchvision.models as models
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import torch.optim as optim


model_pt = models.mobilenet_v2(pretrained=True)

for param in model_pt.parameters():
    param.requires_grad = False

model_pt.classifier[1] = nn.Linear(model_pt.last_channel, 5)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

train_data = datasets.ImageFolder(
    'Training_Image_Folder', transform=train_transform
)

val_data = datasets.ImageFolder(
    'Validation_Image_Folder', transform=val_transform
)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pt.parameters(), lr=1e-4)

for epoch in range(10):
    model_pt.train()

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_pt(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


    print(f'Epoch {epoch+1}, Loss: {loss.item()}')