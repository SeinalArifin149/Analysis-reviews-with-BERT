import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==============================
# 1. LOAD & CLEAN DATA
# ==============================
file_path = "Top Pariwisata Bangkalan (Cleaning).csv"
# file_path = "Top Pariwisata Bangkalan (Cleaning with stopword).csv"
df = pd.read_csv(file_path)

print("Kolom dataset:", df.columns)

# drop data kosong
df = df.dropna(subset=["teks", "stars"])
df["teks"] = df["teks"].astype(str)

# ==============================
# 2. BUAT LABEL DARI STARS
# ==============================
def convert_label(star):
    if star <= 3:
        return 0  # negative
    else:
        return 1  # positive

df["label"] = df["stars"].apply(convert_label)
df["label"] = df["label"].astype(int)

print("\nDistribusi label:")
print(df["label"].value_counts())

# ==============================
# 3. SPLIT DATA
# ==============================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["teks"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# ==============================
# 4. LOAD MODEL
# ==============================
model_name = "indobenchmark/indobert-base-p1"
id2label = {0: "negative", 1: "positive"}
label2id = {label: idx for idx, label in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model_config = AutoConfig.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=model_config,
    ignore_mismatched_sizes=True,
)

# ==============================
# 5. DATASET CLASS
# ==============================
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)

# ==============================
# 6. METRICS (BIAR ADA AKURASI)
# ==============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ==============================
# 7. TRAINING SETUP
# ==============================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,

    eval_strategy="epoch",        # ✅ versi v5
    save_strategy="epoch",        # ✅ simpan tiap epoch
    logging_dir="./logs",
    logging_steps=10,

    load_best_model_at_end=True   # ✅ ambil model terbaik
)

# ==============================
# 8. TRAINER
# ==============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\n🔥 Mulai training...")
trainer.train()

# ==============================
# 9. SAVE MODEL
# ==============================
trainer.save_model("./model_sentiment_binary")
tokenizer.save_pretrained("./model_sentiment_binary")

print("\nModel selesai dilatih")

# ==============================
# 10. PREDICT FUNCTION
# ==============================
label_map = {
    0: "negative",
    1: "positive"
}

def predict(text):
    if not isinstance(text, str):
        text = str(text)

    if text.strip() == "":
        return "negative"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()

    return label_map[pred]

# ==============================
# 11. TEST
# ==============================
print("\n=== TEST ===")
print("1:", predict("tempatnya indah banget"))  # positive
print("2:", predict("banyak begal disini"))     # negative