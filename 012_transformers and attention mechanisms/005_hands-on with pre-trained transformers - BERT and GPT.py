from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import transformers
import accelerate as ac


dataset = load_dataset("stanfordnlp/imdb")


# tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# prepare data for training
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    # logging_dir='./logs',
    logging_steps=10,
    save_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer
)


trainer.train()
results = trainer.evaluate()
print('Evaluation results:', results)




# experiment with GPR
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
input_text = 'Carrick is a class act and he still embodies the values United are meant to represent as a club. Carrick played under Sir Alex Ferguson and is an old-school manager in the sense that he wants to keep everything inside the dressing room. Some fans may want him to rant and rave and show more emotion in public but you are never going to get that from Carrick. When it comes to the media, United are like Real Madrid and Barcelona. Everything that happens at the club is headline news - especially in the social media age. United fans feel like they are under siege. Carrick is on their side. He gives very little away when he speaks in public. He wears his club suit with pride and always says what is in the best interests of the club. He never makes it about himself. The clue is in the name: United.'
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = gpt_model.generate(input_ids, max_length=100, num_return_sequences=1)

print('\n\nGenerate Text: ', tokenizer.decode(
    output[0], skip_special_tokens=True
))