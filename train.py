import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import os
import argparse
from efficia_1.model import Efficia1
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.amp import GradScaler, autocast
import torch.utils.checkpoint as checkpoint
import re
import unicodedata

# --- 1. Konfigurace ---
DIM = 832
DEPTH = 12
HEADS = 8
COMPRESSED_DIM = 416
WINDOW_SIZE = 256
MEM_SIZE = 512
FF_MULT = 4

BATCH_SIZE = 4
SEQ_LEN = 256
EPOCHS = 1
LEARNING_RATE = 1e-5  # Sníženo pro stabilitu
CHECKPOINT_PATH = "efficia1_checkpoint.pth"
DATASET_PATH = "dataset.txt"
TOKENIZER_PATH = "bpe_tokenizer.json"

# --- 2. Zpracování dat ---
def sanitize_text(text):
    """Normalizuje text: zachová emojis, odstraní control characters."""
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_generator(file_path, chunk_size=10**6):
    """Generátor pro čtení souboru po chunkách."""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield sanitize_text(chunk)

def train_tokenizer(file_path, vocab_size=15000):
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading existing tokenizer from {TOKENIZER_PATH}")
        return Tokenizer.from_file(TOKENIZER_PATH)
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]"], min_frequency=2)
    
    tokenizer.train_from_iterator(chunk_generator(file_path), trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer trained and saved to {TOKENIZER_PATH}")
    return tokenizer

def get_text_and_vocab(file_path, use_bpe=True):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = sanitize_text(f.read())
    
    if not text.strip():
        print(f"Error: Dataset file {file_path} is empty after sanitization")
        exit(1)
    
    if use_bpe:
        tokenizer = train_tokenizer(file_path)
        vocab_size = tokenizer.get_vocab_size()
        encoded_len = len(tokenizer.encode(text).ids)
        if encoded_len < SEQ_LEN:
            print(f"Error: Dataset too small ({encoded_len} tokens) for SEQ_LEN={SEQ_LEN}")
            exit(1)
        print(f"Total tokens after BPE tokenization: {encoded_len}")
        return text, tokenizer, vocab_size
    else:
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        char_to_int = {ch: i for i, ch in enumerate(chars)}
        int_to_char = {i: ch for i, ch in enumerate(chars)}
        return text, (char_to_int, int_to_char), vocab_size

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len, use_bpe=True):
        self.seq_len = seq_len
        self.use_bpe = use_bpe
        if use_bpe:
            self.tokenizer = tokenizer
            self.encoded_text = tokenizer.encode(text).ids
        else:
            self.char_to_int = tokenizer[0]
            self.encoded_text = [self.char_to_int[c] for c in text]

    def __len__(self):
        return len(self.encoded_text) - self.seq_len

    def __getitem__(self, idx):
        inputs = torch.tensor(self.encoded_text[idx : idx + self.seq_len], dtype=torch.long)
        targets = torch.tensor(self.encoded_text[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return inputs, targets

# --- 3. Inicializace modelu ---
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Snížena std pro stabilitu
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Snížena std

# --- 4. Tréninková smyčka ---
def train(use_bpe=True):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GiB")

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return

    text, tokenizer, vocab_size = get_text_and_vocab(DATASET_PATH, use_bpe)
    
    fraction = 0.01  # 1 % dat pro rychlý test
    subset_size = int(len(text) * fraction)
    text = text[:subset_size]
    encoded_len = len(tokenizer.encode(text).ids) if use_bpe else len(text)
    print(f"Dataset loaded. Using {fraction*100:.0f}% of data ({encoded_len} tokens). Vocabulary size: {vocab_size}")

    # Debug: Ověření tokenizace
    sample_text = "Beginners BBQ Class Taking Place in Missoula!"
    sample_tokens = tokenizer.encode(sample_text).ids
    print(f"Sample tokens for '{sample_text}': {sample_tokens}")
    print(f"Sample decoded: {tokenizer.decode(sample_tokens)}")

    model = Efficia1(
        num_tokens=vocab_size,
        dim=DIM,
        depth=DEPTH,
        compressed_dim=COMPRESSED_DIM,
        heads=HEADS,
        window_size=WINDOW_SIZE,
        mem_size=MEM_SIZE,
        ff_mult=FF_MULT
    ).to(device)
    
    initialize_weights(model)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params:,} parameters.")
    print(f"GPU memory after model creation: {torch.cuda.memory_allocated(device)/1e9:.2f} GiB")

    dataset = TextDataset(text, tokenizer, SEQ_LEN, use_bpe)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')

    ACCUM_STEPS = 16
    total_loss = 0
    accum_count = 0
    nan_steps = 0

    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        try:
            start_epoch = model.load_checkpoint(CHECKPOINT_PATH, optimizer)
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0

    global_memory = None
    compressed_state = None

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        accum_count = 0
        nan_steps = 0

        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if global_memory is not None:
                if global_memory.size(0) != BATCH_SIZE:
                    global_memory = global_memory.repeat(BATCH_SIZE, 1, 1)
                global_memory = global_memory.detach()

            if compressed_state is not None:
                if compressed_state.size(0) != BATCH_SIZE:
                    compressed_state = compressed_state.repeat(BATCH_SIZE, 1)
                compressed_state = compressed_state.detach()

            with autocast('cuda'):
                logits, global_memory, compressed_state = checkpoint.checkpoint(
                    lambda x, gm, cs: model(x, gm, cs), inputs, global_memory, compressed_state, use_reentrant=False
                )
                logits = torch.clamp(logits, min=-10.0, max=10.0)  # Stabilizace logits
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at step {i+1}, skipping update")
                print(f"Input sample: {inputs[0][:10].tolist()}")
                print(f"Logits max: {torch.max(torch.abs(logits)).item()}")
                nan_steps += 1
                optimizer.zero_grad()
                accum_count = 0
                continue

            if (i + 1) % 5 == 0:
                max_logits = torch.max(torch.abs(logits)).item()
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Max logits: {max_logits:.4f}")

            scaler.scale(loss).backward()
            accum_count += 1

            if accum_count == ACCUM_STEPS or i == len(dataloader) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # Silnější clipping
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accum_count = 0

            total_loss += loss.item()

        avg_loss = total_loss / (len(dataloader) - nan_steps) if len(dataloader) > nan_steps else float('nan')
        print(f"--- End of Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}, NaN steps: {nan_steps} ---")
        print(f"GPU memory after epoch: {torch.cuda.memory_allocated(device)/1e9:.2f} GiB")

        print("Saving checkpoint...")
        model.save_checkpoint(CHECKPOINT_PATH, optimizer, epoch + 1)

    print("Training finished.")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Efficia-1 model")
    parser.add_argument("--use_bpe", action="store_true", default=True, help="Use BPE tokenizer instead of char-level")
    args = parser.parse_args()
    train(use_bpe=args.use_bpe)
