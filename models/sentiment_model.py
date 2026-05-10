import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TextDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
    
class SentimentModel:
    def __init__(self, model_name="indobenchmark/indobert-base-p1"):
        self.model_name = model_name
        self.id2label = {0: "negative", 1: "positive"}
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        
        print(f"🔧 Inisialisasi Tokenizer & Model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_config = AutoConfig.from_pretrained(
            self.model_name, num_labels=2, id2label=self.id2label, label2id=self.label2id
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=self.model_config, ignore_mismatched_sizes=True
        )

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self, df):
        print("\n⚙️ Persiapan Data untuk Training Sentimen...")
        df["label"] = df["stars"].apply(lambda x: 0 if x <= 3 else 1)
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["teks"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
        )

        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )

        print("🔥 Mulai Training IndoBERT...")
        trainer.train()
        
        print("💾 Menyimpan Model Sentimen...")
        trainer.save_model("./model_sentiment_binary")
        self.tokenizer.save_pretrained("./model_sentiment_binary")
        print("✅ Training Sentimen Selesai!\n")

    def predict(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return "negative"
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        pred = torch.argmax(outputs.logits, dim=1).item()
        return self.id2label[pred]