from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
import os

# Disable logs
os.environ["WANDB_DISABLED"] = "true"


# Load Dataset
dataset = load_dataset("stanfordnlp/imdb")

# Use smaller subset for faster training
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(5000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))


# Load Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenization Function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128
    )


# Tokenize Dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)


# Prepare Dataset
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")


# Load Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)


# Train Model
trainer.train()


# 11. Save Model
trainer.save_model("./sentiment-model")
tokenizer.save_pretrained("./sentiment-model")


# Test Prediction
classifier = pipeline(
    "sentiment-analysis",
    model="./sentiment-model"
)

print(classifier("This movie was absolutely amazing!"))