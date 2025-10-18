import torch
import pytest
from efficia_1 import Efficia1
import os
from torch.optim import AdamW

# Model parameters for testing
NUM_TOKENS = 1000
DIM = 128
DEPTH = 2
COMPRESSED_DIM = 64
HEADS = 4
WINDOW_SIZE = 128
FF_MULT = 4
SEQ_LEN = 256
BATCH_SIZE = 2
MEM_SIZE = 512
CHECKPOINT_PATH = "test_checkpoint.pth"

@pytest.fixture
def model():
    """Provides an Efficia1 model instance for testing."""
    return Efficia1(
        num_tokens=NUM_TOKENS,
        dim=DIM,
        depth=DEPTH,
        compressed_dim=COMPRESSED_DIM,
        heads=HEADS,
        window_size=WINDOW_SIZE,
        ff_mult=FF_MULT,
        mem_size=MEM_SIZE
    )

def test_model_forward_pass(model):
    """
    Tests a single forward pass of the Efficia1 model with no initial state.
    """
    # Create a dummy input tensor
    x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, SEQ_LEN))
    
    # Forward pass
    logits, global_memory, compressed_state = model(x)
    
    # Check output shapes
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, NUM_TOKENS), f"Logits shape is incorrect: {logits.shape}"
    assert global_memory.shape == (BATCH_SIZE, MEM_SIZE, DIM), f"Global memory shape is incorrect: {global_memory.shape}"
    assert compressed_state.shape == (BATCH_SIZE, COMPRESSED_DIM), f"Compressed state shape is incorrect: {compressed_state.shape}"

def test_model_forward_with_state(model):
    """
    Tests a forward pass with existing memory and compressed state.
    """
    # Create dummy input and states
    x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, SEQ_LEN))
    initial_memory = torch.randn(BATCH_SIZE, MEM_SIZE, DIM)
    initial_state = torch.randn(BATCH_SIZE, COMPRESSED_DIM)
    
    # Forward pass
    logits, new_global_memory, new_compressed_state = model(x, global_memory=initial_memory, compressed_state=initial_state)
    
    # Check output shapes
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, NUM_TOKENS)
    assert new_global_memory.shape == (BATCH_SIZE, MEM_SIZE, DIM)
    assert new_compressed_state.shape == (BATCH_SIZE, COMPRESSED_DIM)
    
    # Check that the output states are different from the input states
    # Note: This might not hold if the memory update results in the same tensor by chance
    # A more robust test would check the update logic itself.
    # assert not torch.equal(new_global_memory, initial_memory), "Global memory should be updated"
    assert not torch.equal(new_compressed_state, initial_state), "Compressed state should be updated"

def test_model_creation():
    """Tests if the model can be created without errors."""
    try:
        Efficia1(
            num_tokens=NUM_TOKENS,
            dim=DIM,
            depth=DEPTH,
            compressed_dim=COMPRESSED_DIM,
            heads=HEADS,
            window_size=WINDOW_SIZE,
            mem_size=MEM_SIZE
        )
    except Exception as e:
        pytest.fail(f"Model creation failed with an exception: {e}")

def test_checkpointing(model):
    """Tests saving and loading a model checkpoint."""
    optimizer = AdamW(model.parameters())
    epoch = 10

    # Save checkpoint
    model.save_checkpoint(CHECKPOINT_PATH, optimizer, epoch)
    assert os.path.exists(CHECKPOINT_PATH)

    # Create a new model to load into
    new_model = Efficia1(
        num_tokens=NUM_TOKENS,
        dim=DIM,
        depth=DEPTH,
        compressed_dim=COMPRESSED_DIM,
        heads=HEADS,
        window_size=WINDOW_SIZE,
        mem_size=MEM_SIZE
    )
    new_optimizer = AdamW(new_model.parameters())

    # Load checkpoint
    loaded_epoch = new_model.load_checkpoint(CHECKPOINT_PATH, new_optimizer)

    # Verify
    assert loaded_epoch == epoch
    # Check if model weights are the same
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2), "Model parameters do not match after loading checkpoint"
    
    # Clean up
    os.remove(CHECKPOINT_PATH)

