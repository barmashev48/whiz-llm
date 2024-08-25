import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-wikitext2-small")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-wikitext2-small")

# Ensure the model uses the right padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Function to generate text
def generate_text(prompt, max_length=50, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        num_beams=5,
        early_stopping=True,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Example prompt
prompt = "what are you?"

# Generate text
generated_text = generate_text(prompt)

# Print the generated text
for i, text in enumerate(generated_text):
    print(f"Generated Text {i+1}:\n{text}\n")
