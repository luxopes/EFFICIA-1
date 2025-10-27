import torch
import torch.nn.functional as F
from efficia_1 import Efficia1
import os

# --- 1. Konfigurace ---
# Musí být STEJNÁ jako při tréninku
DIM = 128
DEPTH = 4
HEADS = 4
COMPRESSED_DIM = 96
WINDOW_SIZE = 256
MEM_SIZE = 512
FF_MULT = 4

# Parametry generování
PROMPT = "What is AI"
MAX_LEN = 200
TEMPERATURE = 0.8 # Vyšší hodnota = více náhodný text, nižší = více deterministický

# Cesty
CHECKPOINT_PATH = "efficia1_checkpoint.pth"
DATASET_PATH = "dataset.txt"

# --- 2. Pomocné funkce (stejné jako v train.py) ---
def get_text_and_vocab(file_path):
    """Načte text ze souboru a vytvoří znakový slovník."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return vocab_size, char_to_int, int_to_char

# --- 3. Generovací funkce ---
def generate(model, prompt, max_len, temperature, char_to_int, int_to_char, device):
    """Generuje text pomocí natrénovaného modelu."""
    model.eval()
    
    print(f"Prompt: '{prompt}'")
    print("Generování...", end='', flush=True)

    # Inicializace stavů
    global_memory = None
    compressed_state = None
    
    # Zpracování promptu
    prompt_tokens = [char_to_int[c] for c in prompt]
    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # "Zahřátí" modelu - zpracování promptu pro nastavení interních stavů
    if prompt_tensor.size(1) > 0:
        _, global_memory, compressed_state = model(prompt_tensor, global_memory, compressed_state)
        # Vezmeme poslední token z promptu jako startovací pro generování
        current_token = prompt_tensor[:, -1:]
    else:
        # Pokud je prompt prázdný, začneme s náhodným tokenem
        current_token = torch.randint(0, len(char_to_int), (1, 1), device=device)


    generated_text = prompt
    
    with torch.no_grad():
        for _ in range(max_len):
            # Dopředný průchod pro jeden token
            logits, global_memory, compressed_state = model(current_token, global_memory, compressed_state)
            
            # Získání logitů pro poslední token a aplikace teploty
            last_logits = logits[:, -1, :] / temperature
            
            # Aplikace softmax pro získání pravděpodobností
            probs = F.softmax(last_logits, dim=-1)
            
            # Výběr dalšího tokenu na základě pravděpodobností
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Připojení vygenerovaného znaku
            char = int_to_char[next_token.item()]
            generated_text += char
            
            # Nastavení vygenerovaného tokenu jako vstupu pro další krok
            current_token = next_token

    print(" hotovo.")
    return generated_text

# --- 4. Hlavní skript ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Načtení slovníku
    try:
        vocab_size, char_to_int, int_to_char = get_text_and_vocab(DATASET_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

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

    # Načtení checkpointu
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        print("Please train the model first using 'train.py'")
        return
        
    try:
        model.load_checkpoint(CHECKPOINT_PATH)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Generování textu
    generated_text = generate(model, PROMPT, MAX_LEN, TEMPERATURE, char_to_int, int_to_char, device)
    
    print("\n--- Výsledek ---")
    print(generated_text)
    print("----------------")

if __name__ == "__main__":
    main()
