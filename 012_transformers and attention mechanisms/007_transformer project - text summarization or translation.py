from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments



datasets = load_dataset("abisee/cnn_dailymail", "3.0.0")
print(datasets['train'][0])


tokenizer = AutoTokenizer.from_pretrained("t5-small")

# tokenize for summarization
def tokenize_function(examples):
    inputs = [
        'summarize: ' + doc for doc in examples['article']
    ]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # tokenize the targets
    labels = tokenizer(
        text_target = examples['highlights'], max_length=128, truncation=True
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


tokenized_datasets = datasets.map(tokenize_function, batched=True)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

training_args = Seq2SeqTrainingArguments(
    output_dir="output",
    eval_strategy="epoch",
    # save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class = tokenizer
)


trainer.train()

sample_test = "The transformer architecture has revolutionized the field of natural language processing, enabling models to understand and generate human-like text. It utilizes self-attention mechanisms to capture relationships between words in a sequence, allowing for more context-aware representations. This has led to significant advancements in tasks such as machine translation, text summarization, and question answering."
inputs = tokenizer('summarize: ' + sample_test, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)
print('Generated Summary: ', tokenizer.decode(outputs[0], skip_special_tokens=True))


# metric = load_metric('rouge')
# predictions = outputs['generated_text']
# references = dataset['validation']['highlights']
# results = metric.compute(predictions=predictions, references=references)
# print(results)