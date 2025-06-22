import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from tqdm import tqdm
from dataclasses import dataclass

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths to datasets
TRAIN_FILE = "data/train.csv"
DEV_FILE = "data/dev.csv"
TEST_FILE = "data/HornMT.csv"

# Special tokens
BOS, EOS, PAD = 1, 2, 3
batch_size = 32
max_seq_len = 100

# SentencePieceProcessor class remains the same
class SentencePieceProcessor:
    def __init__(self, src_file, trg_file, src_lang="eng", trg_lang="amh", vocab_size=25000):
        self.tokenizers = {}

        if not os.path.exists(f"{src_lang}.model"):
            self.train_sentencepiece(src_file, src_lang, vocab_size)
        if not os.path.exists(f"{trg_lang}.model"):
            self.train_sentencepiece(trg_file, trg_lang, vocab_size)

        self.load_tokenizers()

    def train_sentencepiece(self, input_file, model_prefix, vocab_size):
        spm.SentencePieceTrainer.train(
            f'--input={input_file} --model_prefix={model_prefix} --character_coverage=1.0 --vocab_size={vocab_size} --model_type=bpe'
        )

    def load_tokenizers(self):
        self.tokenizers = {
            "eng": spm.SentencePieceProcessor(model_file="eng.model"),
            "amh": spm.SentencePieceProcessor(model_file="amh.model")
        }

    def tokenize(self, text, lang):
        return self.tokenizers[lang].encode_as_ids(text)

    def detokenize(self, ids, lang):
        return self.tokenizers[lang].decode_ids(ids)

# Modified Dataset class for English to Amharic
class MT_Dataset(Dataset):
    def __init__(self, file_path, tokenizer, is_test=False):
        self.data = pd.read_csv(file_path)
        self.is_test = is_test
        
        required_columns = ["eng"] if is_test else ["eng", "amh"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = str(self.data["eng"].iloc[idx]).strip()
        
        if self.is_test:
            src_tokens = [BOS] + self.tokenizer.tokenize(src_text, 'eng')[:max_seq_len-2] + [EOS]
            return torch.tensor(src_tokens), src_text
        
        trg_text = str(self.data["amh"].iloc[idx]).strip()
        if not src_text or not trg_text:
            return None

        src_tokens = [BOS] + self.tokenizer.tokenize(src_text, 'eng')[:max_seq_len-2] + [EOS]
        trg_tokens = [BOS] + self.tokenizer.tokenize(trg_text, 'amh')[:max_seq_len-2] + [EOS]
        
        return torch.tensor(src_tokens), torch.tensor(trg_tokens)
# Improved padding function with better error handling
def pad_sequence(batch):
    if any(item is None for item in batch):
        batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    if isinstance(batch[0], tuple) and len(batch[0]) == 2:
        if isinstance(batch[0][1], str):  # For test data
            src_seqs, orig_texts = zip(*batch)
            src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD)
            return src_padded, list(orig_texts)
        else:  # For training data
            src_seqs, trg_seqs = zip(*batch)
            src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD)
            trg_padded = nn.utils.rnn.pad_sequence(trg_seqs, batch_first=True, padding_value=PAD)
            return src_padded, trg_padded
    return None

# Improved Transformer with positional encoding and dropout
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = nn.Embedding(input_dim, embed_dim)
        self.decoder = nn.Embedding(output_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=4*embed_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=4*embed_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output_layer = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg=None):
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        src_padding_mask = (src == PAD).to(src.device)
        
        src_emb = self.dropout(self.pos_encoder(self.encoder(src).transpose(0, 1)))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        if trg is None:  # For inference
            return memory
            
        trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        trg_padding_mask = (trg == PAD).to(trg.device)
        
        trg_emb = self.dropout(self.pos_encoder(self.decoder(trg).transpose(0, 1)))
        output = self.transformer_decoder(trg_emb, memory, 
                                       tgt_mask=trg_mask,
                                       tgt_key_padding_mask=trg_padding_mask)
        
        return self.output_layer(output.transpose(0, 1))

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Evaluation function
def evaluate(model, tokenizer, test_dataloader):
    model.eval()
    translations = []
    
    with torch.no_grad():
        for src, orig_texts in tqdm(test_dataloader, desc="Generating Amharic translations"):
            src = src.to(DEVICE)
            memory = model(src)
            
            # Initialize target sequences with BOS token
            trg = torch.ones(src.size(0), 1).fill_(BOS).long().to(DEVICE)
            
            # Generate Amharic text token by token
            for _ in range(max_seq_len - 1):
                output = model(src, trg)
                next_token = output[:, -1, :].argmax(dim=1)
                trg = torch.cat([trg, next_token.unsqueeze(1)], dim=1)
                
                # Stop if all sequences have generated EOS token
                if (next_token == EOS).all():
                    break
            
            # Convert token IDs to Amharic text
            for seq in trg:
                # Remove special tokens (BOS, EOS, PAD)
                token_ids = [id.item() for id in seq if id.item() not in [BOS, EOS, PAD]]
                # Detokenize to Amharic
                translation = tokenizer.detokenize(token_ids, 'amh')
                translations.append(translation)
    
    return translations

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, epochs=10):
    best_val_loss = float('inf')
    
    # Load previous checkpoint if exists
    if os.path.exists("best_checkpoint.pth"):
        model.load_state_dict(torch.load("best_checkpoint.pth"))
        print("Loaded previous best model checkpoint.")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            if batch is None:
                continue
                
            src, trg = batch
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])  # exclude last target token
            
            output = output.contiguous().view(-1, output.size(-1))
            trg = trg[:, 1:].contiguous().view(-1)  # exclude first target token (BOS)
            
            loss = loss_fn(output, trg)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        # Validation
        val_loss = validate(model, val_dataloader, loss_fn)
        
        avg_train_loss = total_loss / max(1, num_batches)  # Avoid division by zero
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_checkpoint.pth")
            print(f"New best model saved with validation loss: {val_loss:.4f}")

        scheduler.step(val_loss)

# Validation function
def validate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
                
            src, trg = batch
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            output = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.size(-1))
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = loss_fn(output, trg)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Main execution
def main():
    # Initialize tokenizer with English as source and Amharic as target
    processor = SentencePieceProcessor("english.txt", "amharic.txt")
    
    # Model parameters
    input_dim = output_dim = 25000  # vocab size
    embed_dim = 512
    model = Transformer(input_dim, output_dim, embed_dim).to(DEVICE)
    
    # Training mode
    if not os.path.exists("best_checkpoint.pth"):
        print("No checkpoint found. Starting training...")
        
        # Create dataloaders for training
        train_dataset = MT_Dataset(TRAIN_FILE, processor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, collate_fn=pad_sequence)
        
        val_dataset = MT_Dataset(DEV_FILE, processor)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                                  shuffle=False, collate_fn=pad_sequence)
        
        # Initialize optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=2, verbose=True)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
        
# Train the model
        # train(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, epochs=30)
    
    # Evaluation mode
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load("best_checkpoint.pth"))
    
    # Create test dataloader
    test_dataset = MT_Dataset(TEST_FILE, processor, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, collate_fn=pad_sequence)
    
    # Generate Amharic translations
    print("Generating Amharic translations...")
    translations = evaluate(model, processor, test_dataloader)
    
    # Save translations to test.csv
    test_df = pd.read_csv(TEST_FILE)
    test_df['translated'] = translations
    test_df.to_csv(TEST_FILE, index=False)
    print(f"Amharic translations saved to {TEST_FILE}")

    # Verify the output
    # print("\nSample translations (first 3 examples):")
    # for i in range(min(3, len(test_df))):
    #     print(f"\nEnglish: {test_df['eng'].iloc[i]}")
    #     print(f"Amharic: {test_df['translated'].iloc[i]}")

if __name__ == "__main__":
    main()