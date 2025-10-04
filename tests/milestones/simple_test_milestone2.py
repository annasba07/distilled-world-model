#!/usr/bin/env python3
"""
Simple Milestone 2 Capability Test: Next-Frame Prediction
Tests temporal prediction capabilities with basic console output
"""

import torch
import torch.nn.functional as F
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Import our models
try:
    from src.models.temporal_predictor import WorldModelWithPrediction, calculate_temporal_metrics
    from src.data.temporal_dataset import TemporalGameDataset
    print("Core modules imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import core modules: {e}")
    print("Please ensure you are running from project root.")
    exit(1)


def test_model_loading():
    """Test 1: Temporal Model Loading & Architecture"""
    print("\n[TEST 1] Temporal Model Loading & Architecture")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Create temporal prediction model
        model = WorldModelWithPrediction(
            in_channels=3,
            latent_dim=256,
            num_embeddings=512,
            d_model=512,
            num_layers=6,
            num_heads=8,
            max_sequence_length=32,
            freeze_vqvae=False
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        vqvae_params = sum(p.numel() for p in model.vqvae.parameters())
        temporal_params = sum(p.numel() for p in model.temporal_model.parameters())
        
        # Calculate model size
        model_size_mb = total_params * 4 / (1024 * 1024)  # FP32
        
        print(f"   [OK] Model loaded successfully")
        print(f"   [INFO] Total parameters: {total_params:,}")
        print(f"   [INFO] VQ-VAE params: {vqvae_params:,}")
        print(f"   [INFO] Temporal params: {temporal_params:,}")
        print(f"   [INFO] Model size: {model_size_mb:.1f} MB")
        
        return {
            'status': 'success',
            'total_parameters': total_params,
            'vqvae_parameters': vqvae_params,
            'temporal_parameters': temporal_params,
            'model_size_mb': model_size_mb
        }
        
    except Exception as e:
        print(f"   [ERROR] Test failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def test_memory_usage():
    """Test 2: Memory Usage for Temporal Sequences"""
    print("\n[TEST 2] Temporal Memory Usage")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("   [SKIP] CUDA not available")
        return {'status': 'skipped', 'reason': 'CUDA not available'}
    
    try:
        model = WorldModelWithPrediction(
            d_model=512,
            num_layers=4,  # Smaller for memory test
            freeze_vqvae=False
        ).to(device)
        
        # Test different sequence lengths
        for seq_len in [4, 8, 16]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create test sequence
            test_frames = torch.randn(1, seq_len, 3, 256, 256).to(device)
            
            # Forward pass
            with torch.no_grad():
                predicted_latents, target_latents = model(test_frames)
            
            # Measure peak memory
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"   [INFO] Seq length {seq_len}: {peak_memory_gb:.2f} GB")
        
        print(f"   [OK] Memory test completed")
        return {
            'status': 'success',
            'max_memory_gb': peak_memory_gb,
            'under_8gb_limit': peak_memory_gb <= 8.0
        }
        
    except Exception as e:
        print(f"   [ERROR] Test failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def test_basic_prediction():
    """Test 3: Basic Temporal Prediction"""
    print("\n[TEST 3] Basic Temporal Prediction")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = WorldModelWithPrediction(
            d_model=256,  # Smaller for testing
            num_layers=3,
            freeze_vqvae=False
        ).to(device)
        
        # Create simple test sequence
        test_frames = torch.randn(1, 8, 3, 256, 256).to(device)
        
        model.eval()
        with torch.no_grad():
            # Get predictions
            predicted_latents, target_latents = model(test_frames)
            
            # Calculate basic metrics
            mse = F.mse_loss(predicted_latents, target_latents)
            psnr = -10 * torch.log10(mse)
            
            print(f"   [INFO] Prediction MSE: {mse:.6f}")
            print(f"   [INFO] Prediction PSNR: {psnr:.2f} dB")
            print(f"   [INFO] Predicted shape: {predicted_latents.shape}")
            print(f"   [INFO] Target shape: {target_latents.shape}")
        
        print(f"   [OK] Basic prediction test completed")
        return {
            'status': 'success',
            'mse': mse.item(),
            'psnr': psnr.item(),
            'prediction_shape': list(predicted_latents.shape),
            'target_shape': list(target_latents.shape)
        }
        
    except Exception as e:
        print(f"   [ERROR] Test failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def test_inference_speed():
    """Test 4: Inference Speed"""
    print("\n[TEST 4] Inference Speed")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = WorldModelWithPrediction(
            d_model=512,
            num_layers=6,
            freeze_vqvae=False
        ).to(device)
        
        model.eval()
        
        # Warm up
        test_frames = torch.randn(1, 8, 3, 256, 256).to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(test_frames)
        
        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                predicted_latents, target_latents = model(test_frames)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"   [INFO] Average time: {avg_time:.1f}±{std_time:.1f} ms")
        print(f"   [INFO] Time per frame: {avg_time/8:.1f} ms")
        
        under_100ms = (avg_time/8) <= 100.0
        print(f"   [INFO] Under 100ms per frame: {under_100ms}")
        
        print(f"   [OK] Inference speed test completed")
        return {
            'status': 'success',
            'avg_time_ms': avg_time,
            'time_per_frame_ms': avg_time/8,
            'under_100ms': under_100ms
        }
        
    except Exception as e:
        print(f"   [ERROR] Test failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def main():
    """Run all Milestone 2 tests"""
    print("=" * 60)
    print("MILESTONE 2 CAPABILITY TESTS: Next-Frame Prediction")
    print("=" * 60)
    
    # Run tests
    results = {}
    results['model_loading'] = test_model_loading()
    results['memory_usage'] = test_memory_usage()
    results['basic_prediction'] = test_basic_prediction()
    results['inference_speed'] = test_inference_speed()
    
    # Calculate summary
    tests_passed = sum(1 for result in results.values() 
                      if result.get('status') == 'success')
    tests_total = len([r for r in results.values() if r.get('status') != 'skipped'])
    
    # Check milestone achievement
    milestone_achieved = (
        results['model_loading'].get('status') == 'success' and
        results['basic_prediction'].get('status') == 'success' and
        results['inference_speed'].get('status') == 'success'
    )
    
    # Memory check
    memory_ok = (results['memory_usage'].get('status') == 'skipped' or 
                results['memory_usage'].get('under_8gb_limit', True))
    
    print("\n" + "=" * 60)
    print("MILESTONE 2 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print(f"Architecture Ready: {'YES' if milestone_achieved else 'NO'}")
    print(f"Memory Efficient: {'YES' if memory_ok else 'NO'}")
    
    if milestone_achieved:
        print("\n[SUCCESS] MILESTONE 2 INFRASTRUCTURE COMPLETE!")
        print("Next step: Train temporal model to achieve prediction accuracy target")
        print("\nTraining commands:")
        print("  python src/train_temporal.py --num_epochs 30")
        print("  python test_milestone2.py --checkpoint checkpoints/temporal/best.pt")
    else:
        print("\n[PENDING] Some tests failed - check implementation")
    
    # Save results
    Path("test_outputs").mkdir(exist_ok=True)
    with open("test_outputs/milestone2_simple_test.json", 'w') as f:
        json.dump({
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'milestone': 'Next-Frame Prediction (Milestone 2)',
            'tests_passed': tests_passed,
            'tests_total': tests_total,
            'milestone_achieved': milestone_achieved,
            'results': results
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to test_outputs/milestone2_simple_test.json")
    return results


if __name__ == "__main__":
    main()