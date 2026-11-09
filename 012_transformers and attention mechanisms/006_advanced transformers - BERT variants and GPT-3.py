from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


datasets = load_dataset("fancyzhx/ag_news")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

# tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

tokenize_datasets = datasets.map(tokenize_function, batched=True)


# prepare dataset
tokenized_datasets = tokenize_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

# training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps = 500
)


trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    processing_class=tokenizer
)


trainer.train()
results = trainer.evaluate()
print('Evaluation Results: ', results)












# using GPT-3 for text classification
import openai
openai.api_key = ""
try:
    # generate text using GPT-3.5 Turbo
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {'role': 'system', 'context': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'write a short story about a robot learning to cook.'}
        ], max_tokens=170, temperature=0.7
    )

    print('Generate Text: \n', response['choices'][0]['message']['content'].strip())

except Exception as e:
    print(f'An Error occured: {e}')