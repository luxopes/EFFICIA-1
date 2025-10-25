import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import os
from efficia_1.model import Efficia1

# --- 1. Konfigurace ---
# Parametry modelu (upraveny pro menší VRAM nároky a rychlejší testování)
DIM = 128
DEPTH = 4
HEADS = 4
COMPRESSED_DIM = 64
WINDOW_SIZE = 256
MEM_SIZE = 512
FF_MULT = 4

# Tréninkové parametry
BATCH_SIZE = 8
SEQ_LEN = 256 # Sníženo pro menší paměťové nároky
EPOCHS = 1
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "efficia1_checkpoint.pth"
DATASET_PATH = "dataset.txt"

# --- 2. Zpracování dat ---
def get_text_and_vocab(file_path):
    """Načte text ze souboru a vytvoří znakový slovník."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return text, chars, vocab_size, char_to_int, int_to_char

class TextDataset(Dataset):
    """Vytvoří dataset z textového souboru."""
    def __init__(self, text, char_to_int, seq_len):
        self.seq_len = seq_len
        self.char_to_int = char_to_int
        self.encoded_text = [self.char_to_int[c] for c in text]

    def __len__(self):
        # Počet možných sekvencí
        return len(self.encoded_text) - self.seq_len

    def __getitem__(self, idx):
        # Vstupní sekvence
        inputs = torch.tensor(self.encoded_text[idx : idx + self.seq_len], dtype=torch.long)
        # Cílová sekvence (posunutá o jeden token)
        targets = torch.tensor(self.encoded_text[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return inputs, targets

# --- 3. Tréninková smyčka ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Načtení dat a vytvoření slovníku
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return
        
    
    text, chars, vocab_size, char_to_int, int_to_char = get_text_and_vocab(DATASET_PATH)
    
    # Použijeme jen 5% dat pro zrychlení
    subset_size = int(len(text) * 0.1)
    text = text[:subset_size]
    
    print(f"Dataset loaded. Using 0.1% of data ({subset_size} characters). Vocabulary size: {vocab_size}")

    # Vytvoření modelu s dynamickou velikostí slovníku
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

    # Dataloader
    dataset = TextDataset(text, char_to_int, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Optimizer a loss funkce
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    # Možnost načíst checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        try:
            start_epoch = model.load_checkpoint(CHECKPOINT_PATH, optimizer)
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0


    # Inicializace stavů
    global_memory = None
    compressed_state = None

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0

        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Důležité: Odpojení stavů od grafu výpočtů z předchozí iterace
            if global_memory is not None:
                # Zajistíme správnou velikost batch dimenze pro stavy
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
            # Logits mají tvar [batch, seq_len, num_tokens], loss je očekává ve tvaru [batch * seq_len, num_tokens]
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            
            # Zpětná propagace
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Ořezání gradientů
            optimizer.step()

            total_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- End of Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f} ---")

        # Uložení checkpointu
        print("Saving checkpoint...")
        model.save_checkpoint(CHECKPOINT_PATH, optimizer, epoch + 1)

    print("Training finished.")

if __name__ == "__main__":
    train()
