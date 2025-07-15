!pip install transformers datasets accelerate
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, Dat
import os
custom_text = """
In a distant land wrapped in eternal mist, there lived a tribe known as the Whisperkeepers. These peo
Kael was not supposed to be the scribe. He was a stable boy, silent and unremarkable. But fate had a
As he traveled through forgotten ruins and across emerald valleys, he met those who feared the scroll
One evening, under the silver light of twin moons, Kael sat with the scroll. He wrote not of war, but
His tale became legend, passed down in stories told by firesides. The scroll now rests in the Hall of
"""
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
# For small RAM, pad tokens (GPT-2 has no padding by default)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
def load_dataset(file_path, tokenizer, block_size=32):
return TextDataset(
tokenizer=tokenizer,
file_path=file_path,
block_size=block_size,
)
train_dataset = load_dataset("dataset/custom.txt", tokenizer)
training_args = TrainingArguments(
output_dir="./gpt2-finetuned",
overwrite_output_dir=True,
num_train_epochs=5,
per_device_train_batch_size=1,
save_steps=500,
save_total_limit=1,
logging_steps=50,
prediction_loss_only=True,
fp16=False, # Use True if using a GPU with half-precision support
import os
os.environ["WANDB_DISABLED"] = "true"
Fine-tune GPT-2
import os
os.environ["WANDB_DISABLED"] = "true" # Disable wandb before training
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
output_dir="./results",
overwrite_output_dir=True,
num_train_epochs=3,
per_device_train_batch_size=1,
save_steps=500,
save_total_limit=2,
prediction_loss_only=True,
)
trainer = Trainer(
model=model,
args=training_args,
data_collator=data_collator,
train_dataset=train_dataset,
)
trainer.train()
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)
# Give your own custom prompt here
prompt = "In the heart of the enchanted forest,"
output = generator(prompt, max_length=150, num_return_sequences=1)
print(output[0]['generated_text'])
from transformers import pipeline
generator = pipeline('text-generation', model="./gpt2-finetuned", tokenizer=tokenizer)
# Give your prompt here
prompt = "In the heart of the Whispering Hollow"
generated = generator(prompt,
max_length=500,
num_return_sequences=1,
do_sample=True,
temperature=0.9,
top_k=50,
top_p=0.95)
print(generated[0]["generated_text"])
with open("generated_story.txt", "w") as f:
f.write(output[0]['generated_text'])
print("Story saved to 'generated_story.txt'")
prompt = "The scroll began to glow in Kael's hands,"
output = generator(prompt, max_length=150, num_return_sequences=1)
print(output[0]['generated_text'])
prompt = "The ancient scroll whispered secrets to Kael,"
output = generator(prompt, max_length=150, num_return_sequences=3)
for i, result in enumerate(output):
print(f"\n--- Version {i+1} ---\n")
print(result['generated_text'])
prompt = "The scroll glowed with ancient power."
output = generator(prompt, max_length=150, num_return_sequences=3)
for i, result in enumerate(output):
print(f"\n--- Story {i+1} ---\n{result['generated_text']}")# PRODIGY_GA_07
