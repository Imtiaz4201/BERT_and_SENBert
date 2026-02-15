import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import re
from models.BERT import BERT
from models.SentenceBERT import SentenceBERT


# ==================== CUSTOM TOKENIZER ====================
class CustomTokenizer:
    def __init__(self, word2id):
        self.word2id = word2id
        self.pad_id = word2id.get("[PAD]", 0)
        self.cls_id = word2id.get("[CLS]", 1)
        self.sep_id = word2id.get("[SEP]", 2)
        self.mask_id = word2id.get("[MASK]", 3)
        self.unk_id = word2id.get("[UNK]", self.pad_id)

    def tokenize(self, text):
        sent = text.lower()
        sent = re.sub(r"[.,!?\\-]", " ", sent)
        tks = sent.split()
        return tks

    def encode(self, text, max_length=128, padding=True, truncation=True):
        tokens = self.tokenize(text)
        if truncation:
            tokens = tokens[: max_length - 2]
        ids = (
            [self.cls_id]
            + [self.word2id.get(tok, self.unk_id) for tok in tokens]
            + [self.sep_id]
        )
        if padding:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids += [self.pad_id] * pad_len
        return ids

    def __call__(
        self, texts, max_length=128, padding=True, truncation=True, return_tensors="pt"
    ):
        batch_ids = []
        for text in texts:
            ids = self.encode(
                text, max_length=max_length, padding=padding, truncation=truncation
            )
            batch_ids.append(ids)
        input_ids = torch.tensor(batch_ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ==================== LOAD MODEL ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "saved_models/sbert_trained.pth"

checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Retrieve word2id and create tokenizer
word2id = checkpoint["word2id"]
tokenizer = CustomTokenizer(word2id)
print(f"Vocabulary size: {len(word2id)}")

# Reconstruct BERT from saved config
bert_config = checkpoint["bert_config"]
bert_model = BERT(
    vocab_size=bert_config["vocab_size"],
    d_model=bert_config["d_model"],
    n_layers=bert_config["n_layers"],
    n_heads=bert_config["n_heads"],
    max_len=bert_config["max_len"],
)
bert_model.load_state_dict(checkpoint["bert_state_dict"])

# Create SentenceBERT
hidden_dim = bert_config["d_model"]
num_classes = checkpoint["num_classes"]
model = SentenceBERT(bert_model, hidden_dim=hidden_dim, num_classes=num_classes)
model.classifier.load_state_dict(checkpoint["sbert_classifier_state_dict"])
model.to(device)
model.eval()

print("Model loaded successfully!")

# ==================== FLASK APP ====================
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    premise = data.get("premise", "").strip()
    hypothesis = data.get("hypothesis", "").strip()

    if not premise or not hypothesis:
        return jsonify({"error": "Please enter both premise and hypothesis"}), 400

    # Tokenize
    premise_enc = tokenizer(
        [premise], max_length=128, padding=True, truncation=True, return_tensors="pt"
    )
    hypothesis_enc = tokenizer(
        [hypothesis], max_length=128, padding=True, truncation=True, return_tensors="pt"
    )

    premise_ids = premise_enc["input_ids"].to(device)
    premise_mask = premise_enc["attention_mask"].to(device)
    hypothesis_ids = hypothesis_enc["input_ids"].to(device)
    hypothesis_mask = hypothesis_enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(premise_ids, premise_mask, hypothesis_ids, hypothesis_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = torch.argmax(logits, dim=1).item()

    labels = ["entailment", "neutral", "contradiction"]
    result = {
        "label": labels[pred_idx],
        "probabilities": {
            "entailment": float(probs[0]),
            "neutral": float(probs[1]),
            "contradiction": float(probs[2]),
        },
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
