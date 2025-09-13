#!/usr/bin/env python3
"""
Simple Test for Milestone 3: Short Sequence Generation
Tests core functionality without Unicode issues
"""

import torch
import numpy as np
from PIL import Image
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.sequence_generator import SequenceGenerator
from data.sequence_dataset import LongSequenceDataset


def test_milestone3_components():
    """Test all Milestone 3 components"""
    print("="*60)
    print("MILESTONE 3 COMPONENT TESTS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test 1: Sequence Generator Model
    print("\n[TEST 1] Sequence Generator Model")
    try:
        model = SequenceGenerator(
            in_channels=3,
            latent_dim=256,
            num_embeddings=512,
            d_model=256,  # Smaller for testing
            num_layers=4,
            num_heads=8,
            max_sequence_length=32,
            freeze_vqvae=False
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   [OK] Model created with {total_params:,} parameters")
        
    except Exception as e:
        print(f"   [ERROR] Model creation failed: {e}")
        return False
    
    # Test 2: Sequence Dataset
    print("\n[TEST 2] Long Sequence Dataset")
    try:
        dataset = LongSequenceDataset(
            sequence_length=16,
            num_sequences=3,
            image_size=(128, 128)
        )
        
        frames, metadata = dataset[0]
        print(f"   [OK] Dataset created: {frames.shape}")
        print(f"   [INFO] Scene type: {metadata['scene_type']}")
        
    except Exception as e:
        print(f"   [ERROR] Dataset creation failed: {e}")
        return False
    
    # Test 3: Initial Frame Generation  
    print("\n[TEST 3] Initial Frame Generation")
    try:
        # Create a simple test frame
        test_frame = Image.new('RGB', (256, 256), (100, 100, 150))
        frame_tensor = torch.from_numpy(np.array(test_frame).astype(np.float32) / 255.0)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
        
        print(f"   [OK] Initial frame created: {frame_tensor.shape}")
        
    except Exception as e:
        print(f"   [ERROR] Initial frame creation failed: {e}")
        return False
    
    # Test 4: Sequence Generation
    print("\n[TEST 4] Sequence Generation")
    try:
        model.eval()
        with torch.no_grad():
            sequences = model.generate_diverse_sequences(
                frame_tensor,
                num_sequences=2,
                sequence_length=8,
                temperature=0.5
            )
        
        if sequences:
            print(f"   [OK] Generated {len(sequences)} sequences")
            for i, seq in enumerate(sequences):
                print(f"   [INFO] Sequence {i+1}: {seq.shape}")
        else:
            print(f"   [WARNING] No sequences generated")
            
    except Exception as e:
        print(f"   [ERROR] Sequence generation failed: {e}")
        return False
    
    # Test 5: Training Forward Pass
    print("\n[TEST 5] Training Forward Pass")
    try:
        # Create batch of sequences
        batch_size = 1
        seq_len = 8
        test_batch = torch.randn(batch_size, seq_len, 3, 256, 256).to(device)
        
        model.train()
        predicted, target, losses = model(test_batch)
        
        print(f"   [OK] Forward pass successful")
        print(f"   [INFO] Predicted: {predicted.shape}")
        print(f"   [INFO] Target: {target.shape}")
        print(f"   [INFO] Additional losses: {list(losses.keys())}")
        
    except Exception as e:
        print(f"   [ERROR] Training forward pass failed: {e}")
        return False
    
    # Test 6: Memory Usage Check
    print("\n[TEST 6] Memory Usage Check")
    try:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Test memory with medium batch
            test_batch = torch.randn(2, 16, 3, 256, 256).to(device)
            
            with torch.no_grad():
                _ = model(test_batch)
            
            memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"   [OK] Memory usage: {memory_gb:.2f} GB")
            
            if memory_gb <= 8.0:
                print(f"   [OK] Under 8GB memory limit")
            else:
                print(f"   [WARNING] Above 8GB memory limit")
        else:
            print(f"   [SKIP] CUDA not available for memory test")
        
    except Exception as e:
        print(f"   [ERROR] Memory test failed: {e}")
        return False
    
    # Test 7: Speed Benchmark
    print("\n[TEST 7] Generation Speed")
    try:
        model.eval()
        
        # Warm up
        with torch.no_grad():
            _ = model.generate_diverse_sequences(frame_tensor, num_sequences=1, sequence_length=5)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            sequences = model.generate_diverse_sequences(
                frame_tensor, 
                num_sequences=1, 
                sequence_length=15,
                temperature=0.8
            )
        end_time = time.time()
        
        generation_time = end_time - start_time
        if sequences and len(sequences[0]) > 0:
            frames_per_second = len(sequences[0]) / generation_time
            print(f"   [OK] Generation time: {generation_time:.2f}s")
            print(f"   [INFO] Speed: {frames_per_second:.1f} frames/second")
        else:
            print(f"   [WARNING] Generation speed test incomplete")
        
    except Exception as e:
        print(f"   [ERROR] Speed test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("MILESTONE 3 TEST SUMMARY")
    print("="*60)
    print("[SUCCESS] All core components working!")
    print("Architecture: Sequence Generator with Enhanced Temporal Model")
    print("Capability: Multi-frame coherent sequence generation")
    print("Status: Ready for training to achieve Milestone 3 targets")
    print("\nNext steps:")
    print("1. Train with: python src/train_sequence.py --num_epochs 40")
    print("2. Test quality metrics and coherence")
    print("3. When targets achieved: Milestone 3 complete!")
    
    return True


if __name__ == "__main__":
    success = test_milestone3_components()
    if success:
        print("\n[FINAL] Milestone 3 infrastructure test: PASSED")
    else:
        print("\n[FINAL] Milestone 3 infrastructure test: FAILED")