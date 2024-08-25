import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load the Mistral-7B tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistral-7b")

# Set the end-of-sequence token as the padding token
tokenizer.pad_token = tokenizer.eos_token

# Load the pre-trained Mistral-7B model
model = AutoModelForCausalLM.from_pretrained("mistral-7b", device_map="auto")

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
    output_dir="./results-mistral",
    evaluation_strategy="epoch",
    learning_rate=2e-5,  
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    gradient_accumulation_steps=16,  
    weight_decay=0.01,
    save_steps=10_000,  
    save_total_limit=2,  
    logging_dir="./logs-mistral",  
    fp16=True,  
    push_to_hub=False,  

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
model.save_pretrained("./whizz-llm")
tokenizer.save_pretrained("./whizz-llm")
