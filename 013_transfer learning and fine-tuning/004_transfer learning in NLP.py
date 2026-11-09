from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration


dataset = load_dataset('stanfordnlp/imdb')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./output',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2
)


trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer
)

trainer.train()
results = trainer.evaluate()
print('Evaluation Results: ', results)
print('\n\n\n')





tokenizer1 = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def preprocess_t5(examples):
    inputs = ['classify sentiment: ' + doc for doc in examples['text']]

    model_inputs = tokenizer1(
        inputs, max_length=128,
        trancation=True, padding='max_length'
    )

    model_inputs['labels'] = tokenizer1(
        examples['labels'], max_length=16,
        truncation=16, padding='max_length'
    )['input_ids']

    return model_inputs

tokenized_t5 = dataset.map(preprocess_t5, batched=True)