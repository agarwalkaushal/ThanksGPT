from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

#Doubt: padding="max_length" and instead use padding=True
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding=True, truncation=True)
    tokens["labels"] = examples["label"]
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text", "intent"])

#Doubt: Increase num_train_epochs=5 or 10.
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()

model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")