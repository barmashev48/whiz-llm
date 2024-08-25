import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the end-of-sequence token as the padding token
tokenizer.pad_token = tokenizer.eos_token

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load the full Wikipedia dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train")

# Preprocess the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

# Apply tokenization to the entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Format the dataset for PyTorch
tokenized_dataset.set_format("torch")

# Prepare for training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_steps=10_000,  # Save model checkpoint every 10,000 steps
    save_total_limit=2,  # Only keep the last 2 checkpoints
    logging_dir="./logs",  # Directory for storing logs
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Fine-tune the model on the full Wikipedia dataset
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./whiz-llm")
tokenizer.save_pretrained("./whiz-llm")
