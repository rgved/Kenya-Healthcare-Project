import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ✅ Step 1: Limit CPU threads to avoid overload
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"

# ✅ Step 2: Load and prepare dataset
df = pd.read_csv("/content/drive/MyDrive/KEN_MODEL/processed_train.csv")

# Rename columns
df = df.rename(columns={
    "Vignette_Clean": "prompt",
    "Clinician_Clean": "summary"
})

# Sanity check
assert "prompt" in df.columns and "summary" in df.columns, "Missing required columns"

# Add instruction
df["prompt"] = df["prompt"] + "\n\nsummarize for clinician use:"

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["prompt", "summary"]])
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ✅ Step 3: Load model and tokenizer
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Important!

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# ✅ Step 4: Apply LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(base_model, peft_config)

# ✅ Step 5: Tokenization
def tokenize_function(batch):
    # Tokenize prompt
    model_inputs = tokenizer(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Tokenize summary (target)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["summary"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# ✅ Step 6: Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    num_train_epochs=3,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    save_total_limit=1,
    remove_unused_columns=False,
    fp16=False,
    report_to="none"
)

# ✅ Step 7: Trainer setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ✅ Step 8: Start training
trainer.train()

# ✅ Step 9: Save final model and tokenizer
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
