"""
Convert Seq2Seq Checkpoint to Encoder/Decoder Format
=====================================================
Splits the flat state dict into separate encoder and decoder dicts
"""

import torch
from collections import OrderedDict

print("=" * 80)
print("CONVERTING CHECKPOINT FORMAT")
print("=" * 80)

# Load original checkpoint
print("\nLoading hindi_spelling_model.pt...")
checkpoint = torch.load('hindi_spelling_model.pt', map_location='cpu', weights_only=False)

print(f"✓ Loaded checkpoint with {len(checkpoint)} keys")

# Split into encoder and decoder
encoder_state = OrderedDict()
decoder_state = OrderedDict()

for key, value in checkpoint.items():
    if key.startswith('encoder.'):
        # Remove 'encoder.' prefix
        new_key = key[8:]  # len('encoder.') = 8
        encoder_state[new_key] = value
    elif key.startswith('decoder.'):
        # Remove 'decoder.' prefix
        new_key = key[8:]  # len('decoder.') = 8
        decoder_state[new_key] = value
    else:
        print(f"Warning: Unknown key {key}")

print(f"\n✓ Split into:")
print(f"  Encoder: {len(encoder_state)} parameters")
print(f"  Decoder: {len(decoder_state)} parameters")

# Create new checkpoint format
new_checkpoint = {
    'encoder': encoder_state,
    'decoder': decoder_state
}

# Save converted checkpoint
output_file = 'hindi_spelling_model_split.pt'
print(f"\nSaving to {output_file}...")
torch.save(new_checkpoint, output_file)

print(f"✓ Saved converted checkpoint")

# Verify the new format
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

verify = torch.load(output_file, map_location='cpu', weights_only=False)
print(f"\nNew checkpoint keys: {list(verify.keys())}")
print(f"Encoder keys (first 5): {list(verify['encoder'].keys())[:5]}")
print(f"Decoder keys (first 5): {list(verify['decoder'].keys())[:5]}")

print("\n" + "=" * 80)
print("CONVERSION COMPLETE!")
print("=" * 80)
print(f"\nNow update hybrid_spell_checker.py to use:")
print(f"  model_path='hindi_spelling_model_split.pt'")
print("=" * 80)
