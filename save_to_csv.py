import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ✅ Paths
model_path = "/content/drive/MyDrive/KEN_MODEL/final_model"
test_path = "/content/drive/MyDrive/KEN_MODEL/processed_test.csv"

# ✅ Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

# ✅ Load and prepare test data
df = pd.read_csv(test_path)
df = df[["Master_Index", "Vignette_Clean"]].copy()

# ✅ Clean vignette text
def clean_vignette(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()

    # Remove bios and first-person references
    lower = text.lower()
    if "i am a nurse" in lower and "kenya" in lower:
        parts = lower.split("in kenya", 1)
        text = parts[-1].strip().capitalize() if len(parts) > 1 else text
    if text.lower().startswith("i "):
        text = text[text.find(" "):].strip()
    return text

df["Cleaned_Vignette"] = df["Vignette_Clean"].apply(clean_vignette)

# ✅ Prompt formatter
def create_prompt(vignette):
    if not vignette:
        return None
    return (
        f"Summarize the following clinical vignette in a formal, concise style:\n\n"
        f"{vignette.strip()}\n\n"
        "Summary:"
    )



df["formatted_prompt"] = df["Cleaned_Vignette"].apply(create_prompt)
df["formatted_prompt"] = df["formatted_prompt"].fillna("Summarize this patient vignette:\n\nNone provided.\n\nSummary:")


# ✅ Post-processing function
def postprocess_summary(text):
    # Remove generic or casual phrases
    for phrase in ["this is a case of", "in this case", "the patient is a", "vignette:", "so we received"]:
        if text.lower().startswith(phrase):
            text = text[len(phrase):].strip()

    # Remove trailing "questions", numbers, and incomplete thoughts
    text = re.sub(r"questions?\s*\d*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"how do i.*", "", text, flags=re.IGNORECASE)

    # Normalize spaces
    text = " ".join(text.split())

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    # Ensure proper ending punctuation
    if text and text[-1] not in ".!?":
        text += "."

    return text.strip()

# ✅ Generate predictions
summaries = []
for prompt in df["formatted_prompt"]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=300,
            num_beams=6,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.2
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_output = postprocess_summary(decoded.strip())
    summaries.append(cleaned_output)

# ✅ Save output
df["Clinician_Clean"] = summaries
df[["Master_Index", "Clinician_Clean"]].to_csv("submission.csv", index=False)
print("✅ submission.csv saved successfully!")

# ✅ Optional: show one example
sample_idx = 0
print("\n🧾 PROMPT:\n", df['formatted_prompt'].iloc[sample_idx])
print("\n📋 GENERATED SUMMARY:\n", df['Clinician_Clean'].iloc[sample_idx])
