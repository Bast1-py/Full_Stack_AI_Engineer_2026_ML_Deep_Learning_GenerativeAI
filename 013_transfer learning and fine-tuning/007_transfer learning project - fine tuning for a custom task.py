from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.lite.python.schema_py_generated import ModelT
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments



dataset = load_dataset('amazon_polarity')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def preprocess_data(examples):
    return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(preprocess_data, batched=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    training_dataset = tokenized_datasets['traing'],
    eval_dataset = tokenized_datasets['test'],
    tokenizer=tokenizer
)

trainer.train()
results = trainer.evaluate()

print('Evaluation Results: ', results)



datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_data = datagen.flow_from_directory(
    'PATH To Dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
)

val_data = datagen.flow_from_directory(
    'PATH To Dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
)

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation='sigmoid')(x)
model = Model(
    inputs=base_model.input,
    outputs=output
)
model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy', metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

val_loss, val_accuracy = model.evaluate()
print(f"Validation Accuracy {val_accuracy}")