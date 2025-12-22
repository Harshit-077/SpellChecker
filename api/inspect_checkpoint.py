"""
Inspect Model Checkpoint Format
================================
Checks what's inside hindi_spelling_model.pt
"""

import torch

print("=" * 80)
print("INSPECTING MODEL CHECKPOINT")
print("=" * 80)

try:
    checkpoint = torch.load('hindi_spelling_model.pt', map_location='cpu', weights_only=False)
    
    print(f"\nCheckpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")
                print(f"  Keys (first 5): {list(value.keys())[:5]}")
            else:
                print(f"\n{key}: {type(value)}")
    else:
        print("\nCheckpoint is a raw state dict (not a dictionary with keys)")
        print(f"State dict keys (first 10): {list(checkpoint.keys())[:10]}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    if isinstance(checkpoint, dict):
        if 'encoder' in checkpoint and 'decoder' in checkpoint:
            print("\n✓ Format: Separate encoder/decoder dictionaries")
            print("✓ This format is COMPATIBLE with hybrid_spell_checker.py")
            print("\nEncoder keys (first 5):", list(checkpoint['encoder'].keys())[:5])
            print("Decoder keys (first 5):", list(checkpoint['decoder'].keys())[:5])
        elif 'model_state_dict' in checkpoint:
            print("\n✗ Format: Single model state dict")
            print("✗ This format is NOT compatible with separate encoder/decoder")
            print("\nYou need to retrain or extract encoder/decoder from Seq2Seq model")
        else:
            print("\n? Unknown format with keys:", list(checkpoint.keys()))
    else:
        print("\n✗ Format: Raw state dict (Seq2Seq combined)")
        print("✗ This needs to be split into encoder/decoder")
        
except Exception as e:
    print(f"\n✗ Error loading checkpoint: {e}")

print("\n" + "=" * 80)
