import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import os
from glob import glob
from efficia_1.model import Efficia1

# --- 1. Model configuration ---
DIM = 384              
DEPTH = 8              
HEADS = 8              
COMPRESSED_DIM = 192   
WINDOW_SIZE = 256
MEM_SIZE = 1024        
FF_MULT = 6            

# --- 2. Training ---
BATCH_SIZE = 16         
SEQ_LEN = 1024         
EPOCHS = 1             
LEARNING_RATE = 2e-4   
CHECKPOINT_PATH = "efficia1_checkpoint_large.pth"
DATASET_PATH = "dataset.txt"
MAX_CKPTS = 3          

# --- 2. Zpracování dat ---
def get_text_and_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    return text, chars, vocab_size, char_to_int, int_to_char

class TextDataset(Dataset):
    def __init__(self, text, char_to_int, seq_len):
        self.seq_len = seq_len
        self.char_to_int = char_to_int
        self.encoded_text = [self.char_to_int[c] for c in text]

    def __len__(self):
        return len(self.encoded_text) - self.seq_len

    def __getitem__(self, idx):
        inputs = torch.tensor(self.encoded_text[idx: idx + self.seq_len], dtype=torch.long)
        targets = torch.tensor(self.encoded_text[idx + 1: idx + self.seq_len + 1], dtype=torch.long)
        return inputs, targets

# --- Pomocná funkce pro mazání starých checkpointů ---
def manage_checkpoints(pattern="checkpoint_step_*.pth", max_ckpts=3):
    files = sorted(glob(pattern), key=os.path.getmtime)
    while len(files) > max_ckpts:
        os.remove(files[0])
        files.pop(0)

# --- 3. Tréninková smyčka ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return

    text, chars, vocab_size, char_to_int, int_to_char = get_text_and_vocab(DATASET_PATH)

    # Použijeme jen 10% dat pro rychlý test
    subset_size = int(len(text) * 0.1)
    text = text[:subset_size]

    print(f"Dataset loaded. Using 10% of data ({subset_size} characters). Vocabulary size: {vocab_size}")

    # Vytvoření modelu
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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params:,} parameters.")

    dataset = TextDataset(text, char_to_int, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    steps_per_epoch = len(dataloader)
    total_steps = EPOCHS * steps_per_epoch

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

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

        for i, (inputs, targets) in enumerate(dataloader):
            step = epoch * steps_per_epoch + i + 1  # globální krok
            inputs, targets = inputs.to(device), targets.to(device)

            # Odpojení předchozích stavů
            if global_memory is not None:
                if global_memory.size(0) != BATCH_SIZE:
                    global_memory = global_memory.repeat(BATCH_SIZE, 1, 1)
                global_memory = global_memory.detach()

            if compressed_state is not None:
                if compressed_state.size(0) != BATCH_SIZE:
                    compressed_state = compressed_state.repeat(BATCH_SIZE, 1)
                compressed_state = compressed_state.detach()

            # Dopředný průchod
            logits, global_memory, compressed_state = model(inputs, global_memory, compressed_state)

            # Výpočet loss
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Logování
            if step % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{steps_per_epoch}], "
                      f"Global Step [{step}/{total_steps}], Loss: {loss.item():.4f}")

            # 💾 Checkpoint každých 10 000 kroků
            if step % 10000 == 0:
                ckpt_path = f"checkpoint_step_{step}.pth"
                print(f"Saving checkpoint at step {step} → {ckpt_path}")
                model.save_checkpoint(ckpt_path, optimizer, epoch + 1)
                manage_checkpoints(max_ckpts=MAX_CKPTS)

        avg_loss = total_loss / len(dataloader)
        print(f"--- End of Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f} ---")

        # Uložení epoch checkpointu
        print("Saving epoch checkpoint...")
        model.save_checkpoint(CHECKPOINT_PATH, optimizer, epoch + 1)

    print("Training finished.")

if __name__ == "__main__":
    train()
