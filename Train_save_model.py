import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ✅ Optional: Limit CPU threads to avoid overheating
torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"

# ✅ Step 1: Load and prepare dataset
df = pd.read_csv("processed_train.csv")
df = df.rename(columns={
    "Vignette_Clean": "prompt",
    "Clinician_Clean": "summary"
})
df["prompt"] = df["prompt"] + "\n\nsummarize for clinician use:"

dataset = Dataset.from_pandas(df[["prompt", "summary"]])
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ✅ Step 2: Load model and tokenizer
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Important for PEFT

model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# ✅ Step 3: Apply LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, peft_config)

# ✅ Step 4: Tokenization function (correctly sets input_ids and labels)
def tokenize_function(batch):
    model_inputs = tokenizer(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    labels = tokenizer(
        batch["summary"],
        padding="max_length",
        truncation=True,
        max_length=128  # summary can be shorter
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs



# ✅ Step 5: Apply tokenization
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names  # ← remove 'prompt' and 'summary'
)

tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names  # ← same here
)


# ✅ Step 6: Training arguments for CPU
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    num_train_epochs=5,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    save_total_limit=1,
    remove_unused_columns=False,
    fp16=False,
    bf16=False,
    report_to="none",  # Disable W&B if not needed
    label_smoothing_factor=0.1
)

# ✅ Step 7: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=lambda data: tokenizer.pad(data, return_tensors="pt"),
)


# ✅ Step 8: Start training
trainer.train()

# ✅ Step 9: Save final model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
