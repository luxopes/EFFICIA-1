# EFFICIA-1: An Experimental, Efficient Language Model Architecture

EFFICIA-1 is a proof-of-concept language model designed for computational efficiency and effective long-context handling. This project provides a complete, working implementation of the EFFICIA-1 architecture in PyTorch, from the core building blocks to training and inference scripts.

The primary goal of this architecture is to explore alternatives to the standard Transformer model that are less computationally intensive (linear complexity) while still retaining powerful context management capabilities.

## Key Architectural Concepts

The EFFICIA-1 architecture is a unique hybrid model that processes information through four distinct stages within each block:

1.  **Local Context Processor (LCP):**
    *   **Purpose:** To efficiently process local context (e.g., nearby words).
    *   **Mechanism:** Uses **windowed linear attention**. Instead of full quadratic attention, it processes the input in overlapping windows, making it significantly faster and less memory-intensive (`O(n)` complexity).

2.  **Global Memory Gate (GMG):**
    *   **Purpose:** To access and integrate long-term memories.
    *   **Mechanism:** Employs **cross-attention**. The current sequence queries a persistent "global memory" tensor, allowing the model to retrieve relevant information from the distant past. This memory is updated using a FIFO (First-In, First-Out) buffer strategy.

3.  **Context Compression Unit (CCU):**
    *   **Purpose:** To compress the current context into a compact state vector.
    *   **Mechanism:** Uses a **Gated Recurrent Unit (GRU) cell** combined with a learned pooling mechanism. This creates a summary of the current sequence, which is passed to subsequent blocks and can be used to maintain state over time.

4.  **Feedback Fusion (FF):**
    *   **Purpose:** To intelligently combine the outputs from the local, global, and compressed contexts.
    *   **Mechanism:** A gating mechanism weighs the information from the LCP, GMG, and the projected CCU state, allowing the model to decide which information stream is most important at any given moment.

## Project Structure

```
efficia-1/
├── src/
│   └── efficia_1/
│       ├── layers/
│       │   ├── common.py       # Core components like RMSNorm, FeedForward
│       │   ├── lcp.py          # Local Context Processor
│       │   ├── gmg.py          # Global Memory Gate
│       │   ├── ccu.py          # Context Compression Unit
│       │   └── ff.py           # Feedback Fusion
│       └── model.py            # Main Efficia1 model assembly
├── tests/
│   └── test_model.py           # Pytest tests for the model
├── train.py                    # Script for training the model
├── infer.py                    # Script for generating text with a trained model
├── requirements.txt            # Project dependencies
├── setup.py                    # Package setup script
├── dataset.txt                 # Example dataset file
└── README.md                   # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/luxopes/EFFICIA-1.git
    cd efficia-1
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the package in editable mode:**
    This step is crucial as it allows the training and inference scripts to find the `efficia_1` module.
    ```bash
    pip install -e .
    ```

## Usage

### 1. Training

The `train.py` script handles the training process. It will automatically use the `dataset.txt` file in the root directory.

To start training, simply run:
```bash
python train.py
```

The script will:
- Build the vocabulary from the dataset.
- Initialize the EFFICIA-1 model.
- Start the training loop.
- Periodically print the loss and save a checkpoint file (`efficia1_checkpoint.pth`) at the end of each epoch.

You can configure model and training parameters (like `DIM`, `DEPTH`, `BATCH_SIZE`, `EPOCHS`, etc.) at the top of the `train.py` file.

### 2. Inference

The `infer.py` script loads a trained checkpoint and generates text from a given prompt.

To run inference:
```bash
python infer.py
```

The script will:
- Load the model architecture with the same parameters used for training.
- Load the weights from `efficia1_checkpoint.pth`.
- Generate text based on the `PROMPT` variable defined inside the script.

You can change the `PROMPT`, `MAX_LEN` (maximum length of generated text), and `TEMPERATURE` (randomness of the output) at the top of the `infer.py` file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
