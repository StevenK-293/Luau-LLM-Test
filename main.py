import os
import math
import random
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class SimpleSubwordTokenizer:
    def __init__(self, vocab_size=5000, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<eos>", "<unk>", "User:", "Assistant:"]
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.vocab = {token: idx for idx, token in enumerate(special_tokens)}
        self.rev_vocab = {idx: token for token, idx in self.vocab.items()}

    def train(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        most_common = counter.most_common(self.vocab_size - len(self.special_tokens))
        for token, _ in most_common:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.rev_vocab[idx] = token

    def encode(self, text):
        tokens = text.split()
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab["<unk>"])
        ids.append(self.vocab["<eos>"])
        return ids

    def decode(self, ids):
        tokens = [self.rev_vocab.get(i, "<unk>") for i in ids]
        tokens = [t for t in tokens if t not in ["<eos>", "<pad>"]]
        return " ".join(tokens)

    @property
    def pad_token_id(self):
        return self.vocab["<pad>"]

    @property
    def eos_token_id(self):
        return self.vocab["<eos>"]

# =============================================================================
# Dataset for Code Samples
# =============================================================================

class CodeDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids += [tokenizer.pad_token_id] * (max_length - len(ids))
            self.data.append(torch.tensor(ids, dtype=torch.long))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# =============================================================================
# Transformer Model for Code Generation (LuauTransformer)
# =============================================================================

class LuauTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, 
                 dim_feedforward, dropout, max_seq_length):
        super(LuauTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Positional encoding: shape [max_seq_length, 1, embedding_dim]
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, 1, embedding_dim))
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc_out.weight, mean=0, std=0.1)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0)
    
    def forward(self, src, tgt):
        # src and tgt: shape [seq_length, batch_size]
        src_emb = self.embedding(src) + self.positional_encoding[:src.size(0), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:tgt.size(0), :]
        transformer_out = self.transformer(src_emb, tgt_emb)
        logits = self.fc_out(transformer_out)
        return logits

# =============================================================================
# Generation Function (with Temperature Sampling)
# =============================================================================

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, device="cpu"):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    generated = input_ids.copy()
    with torch.no_grad():
        for _ in range(max_length - len(generated)):
            inp = torch.tensor(generated).unsqueeze(1).to(device)  # shape: [seq_len, 1]
            output = model(inp, inp)
            logits = output[-1, 0]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated)

# =============================================================================
# Training Pipeline with TensorBoard Logging and Checkpoint Saving
# =============================================================================

def train_model(model, train_loader, val_loader, epochs, learning_rate, device, checkpoint_dir="checkpoints"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.to(device)
    writer = SummaryWriter()
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)  # shape: [batch_size, seq_length]
            src = batch[:, :-1].transpose(0, 1)  # shape: [seq_len-1, batch_size]
            tgt = batch[:, 1:].transpose(0, 1)   # shape: [seq_len-1, batch_size]
            optimizer.zero_grad()
            output = model(src, src)
            loss = criterion(output.view(-1, model.fc_out.out_features), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    src = batch[:, :-1].transpose(0, 1)
                    tgt = batch[:, 1:].transpose(0, 1)
                    output = model(src, src)
                    loss = criterion(output.view(-1, model.fc_out.out_features), tgt.reshape(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        # Generate a sample output for inspection.
        sample_prompt = "function greet(name)"
        sample_output = generate_text(model, tokenizer, sample_prompt, max_length=100, temperature=0.8, device=device)
        writer.add_text("Sample Generation", f"Prompt: {sample_prompt}\nOutput: {sample_output}", epoch)
        print(f"Sample Generation at Epoch {epoch+1}:\n{sample_output}\n")

        # Save a checkpoint.
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}\n")
    
    writer.close()

# =============================================================================
# Chat Interface for Interactive Generation
# =============================================================================

def chat_interface(model, tokenizer, device="cpu"):
    conversation = []
    print("Start chatting with the model! Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        conversation.append("User: " + user_input)
        prompt = "\n".join(conversation) + "\nAssistant:"
        response = generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device=device)
        print("Assistant:", response)
        conversation.append("Assistant: " + response)

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Example basic code snippets
    basic_code_texts = [
        "function add(a, b)\n  return a + b\nend",
        "for i = 1, 10 do\n  print(i)\nend",
        "if x > 0 then\n  print('positive')\nelse\n  print('non-positive')\nend",
        "local table = {1, 2, 3, 4}\nfor _, v in ipairs(table) do\n  print(v)\nend",
        "function factorial(n)\n  if n <= 1 then return 1 end\n  return n * factorial(n - 1)\nend"
    ]
    
    # train the tokenizer.
    tokenizer = SimpleSubwordTokenizer(vocab_size=5000)
    tokenizer.train(basic_code_texts)
 
    train_dataset = CodeDataset(basic_code_texts, tokenizer, max_length=100)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # If you have validation data, create a val_loader. Otherwise, set val_loader to None.
    val_loader = None  # or DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Hyperparameters for the transformer model.
    embedding_dim = 128
    nhead = 8
    num_layers = 4
    dim_feedforward = 256
    dropout = 0.1
    max_seq_length = 100
    
    model = LuauTransformer(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=embedding_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_length=max_seq_length
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_model(model, train_loader, val_loader, epochs=20, learning_rate=0.001, device=device)
    
    prompt = "function greet(name)"
    generated_code = generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device=device)
    print("\nGenerated Code:\n", generated_code)

    # chat_interface(model, tokenizer, device=device)
