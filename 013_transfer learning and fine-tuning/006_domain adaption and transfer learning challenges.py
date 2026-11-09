from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

import random


dataset = load_dataset('pubmed_rct', '20k_rct')
print(dataset['train'][0])

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(preprocess_data, batched=True)
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')


# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=5)

training_args = TrainingArguments(
    output_dir = './result',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,

)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['validation'],
    tokenizer = tokenizer
)

trainer.train()

results = trainer.evaluate()
print('Evaluation Results: ', results)


def augment_text(text):
    synonyms = {
        'cancer': [
            'tumor',
            'malignancy'
        ],
        'study': [
            'research',
            'experiment'
        ]
    }

    new_words = [
        random.choice(synonyms[word]) if word in synonyms else word for word in words
    ]

    return " ".join(new_words)


augmented_data = [
    augment_text(sample['text']) for sample in dataset['train']
]

print(augmented_data[:5])