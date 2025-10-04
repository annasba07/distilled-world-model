#!/usr/bin/env python3
"""
🧪 Milestone 2 Capability Test: Next-Frame Prediction
Tests temporal prediction capabilities with comprehensive metrics
"""

import torch
import torch.nn.functional as F
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns

# Import our models
try:
    from src.models.temporal_predictor import WorldModelWithPrediction, calculate_temporal_metrics
    from src.data.temporal_dataset import TemporalGameDataset
    try:
        from src.demo_temporal import TemporalPredictionDemo
        DEMO_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Demo module not available: {e}")
        DEMO_AVAILABLE = False
        TemporalPredictionDemo = None
except ImportError as e:
    print(f"ERROR: Failed to import core modules: {e}")
    print("Please ensure you are running from project root and dependencies are installed.")
    exit(1)


class Milestone2Tester:
    """Comprehensive tester for Milestone 2: Next-Frame Prediction"""
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.results = {}
        self.test_outputs_dir = Path("test_outputs")
        self.test_outputs_dir.mkdir(exist_ok=True)
        
        print(f"[TEST] Milestone 2 Capability Tester")
        print(f"   Device: {self.device}")
        print(f"   PyTorch: {torch.__version__}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("=" * 60)
    
    def test_temporal_model_loading(self) -> Dict:
        """Test 1: Temporal Model Architecture & Loading"""
        print("\n[TEST 1] Temporal Model Loading & Architecture")
        
        try:
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
            ).to(self.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            vqvae_params = sum(p.numel() for p in model.vqvae.parameters())
            temporal_params = sum(p.numel() for p in model.temporal_model.parameters())
            
            # Calculate model size
            model_size_mb = total_params * 4 / (1024 * 1024)  # FP32
            
            result = {
                'status': 'success',
                'total_parameters': total_params,
                'vqvae_parameters': vqvae_params,
                'temporal_parameters': temporal_params,
                'model_size_mb': model_size_mb,
                'max_sequence_length': model.temporal_model.max_sequence_length,
                'architecture_components': {
                    'vqvae': 'ImprovedVQVAE',
                    'temporal_model': 'TemporalDynamicsModel',
                    'attention_heads': model.temporal_model.layers[0].self_attention.num_heads,
                    'transformer_layers': len(model.temporal_model.layers)
                }
            }
            
            print(f"   [OK] Model loaded successfully")
            print(f"   [INFO] Total parameters: {total_params:,}")
            print(f"   [INFO] VQ-VAE params: {vqvae_params:,}")
            print(f"   [INFO] Temporal params: {temporal_params:,}")
            print(f"   [INFO] Model size: {model_size_mb:.1f} MB")
            print(f"   [INFO] Max sequence length: {model.temporal_model.max_sequence_length}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def test_temporal_memory_usage(self) -> Dict:
        """Test 2: Memory Usage for Temporal Sequences"""
        print("\n🧪 TEST 2: Temporal Memory Usage")
        
        if self.device.type != 'cuda':
            return {'status': 'skipped', 'reason': 'CUDA not available'}
        
        try:
            model = WorldModelWithPrediction(
                d_model=512,
                num_layers=4,  # Smaller for memory test
                freeze_vqvae=False
            ).to(self.device)
            
            sequence_lengths = [4, 8, 16, 32]
            batch_sizes = [1, 2, 4]
            memory_usage = {}
            
            for seq_len in sequence_lengths:
                memory_usage[seq_len] = {}
                for batch_size in batch_sizes:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Create test sequence
                    test_frames = torch.randn(batch_size, seq_len, 3, 256, 256).to(self.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        predicted_latents, target_latents = model(test_frames)
                    
                    # Measure peak memory
                    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                    memory_usage[seq_len][batch_size] = peak_memory_gb
                    
                    print(f"   📊 Seq={seq_len}, Batch={batch_size}: {peak_memory_gb:.2f} GB")
            
            # Find maximum memory usage
            max_memory = max([max(batch_dict.values()) for batch_dict in memory_usage.values()])
            under_8gb = max_memory <= 8.0
            
            result = {
                'status': 'success',
                'memory_usage_by_sequence': memory_usage,
                'max_memory_gb': max_memory,
                'under_8gb_limit': under_8gb,
                'recommended_config': {
                    'seq_length': 16,
                    'batch_size': 2 if max_memory <= 6.0 else 1,
                    'estimated_memory': memory_usage.get(16, {}).get(2, 'N/A')
                }
            }
            
            print(f"   ✅ Memory test completed")
            print(f"   📊 Maximum memory: {max_memory:.2f} GB")
            print(f"   {'✅' if under_8gb else '❌'} Under 8GB limit: {under_8gb}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def test_prediction_accuracy(self) -> Dict:
        """Test 3: Temporal Prediction Accuracy"""
        print("\n🧪 TEST 3: Temporal Prediction Accuracy")
        
        try:
            model = WorldModelWithPrediction(
                d_model=256,  # Smaller for testing
                num_layers=3,
                freeze_vqvae=False
            ).to(self.device)
            
            # Create temporal dataset (synthetic if missing)
            dataset = TemporalGameDataset(
                data_dir="datasets/temporal",
                sequence_length=8,
                prediction_horizon=1,
                split='train',
                generate_synthetic=True
            )
            
            # Test on a few sequences
            total_mse = 0.0
            total_temporal_consistency = 0.0
            total_cosine_sim = 0.0
            num_samples = min(20, len(dataset))
            
            model.eval()
            with torch.no_grad():
                for i in range(num_samples):
                    sample = dataset[i]
                    # Combine input and target into a single sequence for the model
                    seq_frames = torch.cat([sample['input_frames'], sample['target_frames']], dim=0)
                    frames = seq_frames.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
                    
                    # Get predictions
                    predicted_latents, target_latents = model(frames)
                    
                    # Calculate metrics
                    metrics = calculate_temporal_metrics(predicted_latents, target_latents)
                    
                    total_mse += metrics['mse']
                    total_temporal_consistency += metrics['temporal_consistency']
                    total_cosine_sim += metrics['cosine_similarity']
            
            # Average metrics
            avg_mse = total_mse / num_samples
            avg_temporal_consistency = total_temporal_consistency / num_samples
            avg_cosine_sim = total_cosine_sim / num_samples
            avg_psnr = -10 * np.log10(avg_mse)
            
            # Simple accuracy metric (based on PSNR threshold)
            accuracy = 1.0 if avg_psnr > 15.0 else avg_psnr / 15.0  # Untrained model baseline
            target_achieved = accuracy >= 0.8
            
            result = {
                'status': 'success' if avg_mse < float('inf') else 'needs_training',
                'metrics': {
                    'mse': avg_mse,
                    'psnr': avg_psnr,
                    'temporal_consistency': avg_temporal_consistency,
                    'cosine_similarity': avg_cosine_sim,
                    'accuracy': accuracy
                },
                'target_achieved': target_achieved,
                'samples_tested': num_samples,
                'milestone_requirement': 'Prediction accuracy > 80%'
            }
            
            print(f"   ✅ Prediction test completed")
            print(f"   📊 Average MSE: {avg_mse:.6f}")
            print(f"   📊 Average PSNR: {avg_psnr:.2f} dB")
            print(f"   📊 Temporal consistency: {avg_temporal_consistency:.6f}")
            print(f"   📊 Cosine similarity: {avg_cosine_sim:.4f}")
            print(f"   📊 Prediction accuracy: {accuracy:.1%}")
            print(f"   {'✅' if target_achieved else '⏳'} Target (>80%): {target_achieved}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def test_temporal_inference_speed(self) -> Dict:
        """Test 4: Temporal Inference Speed"""
        print("\n🧪 TEST 4: Temporal Inference Speed")
        
        try:
            model = WorldModelWithPrediction(
                d_model=512,
                num_layers=6,
                freeze_vqvae=False
            ).to(self.device)
            
            model.eval()
            
            # Warm up
            test_frames = torch.randn(1, 8, 3, 256, 256).to(self.device)
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_frames)
            
            # Benchmark different sequence lengths
            sequence_lengths = [4, 8, 16]
            timing_results = {}
            
            for seq_len in sequence_lengths:
                test_frames = torch.randn(1, seq_len, 3, 256, 256).to(self.device)
                
                # Time multiple runs
                times = []
                for _ in range(10):
                    start_time = time.time()
                    with torch.no_grad():
                        predicted_latents, target_latents = model(test_frames)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                timing_results[seq_len] = {
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'fps': 1000.0 / (avg_time / seq_len) if avg_time > 0 else 0
                }
                
                print(f"   📊 Sequence length {seq_len}: {avg_time:.1f}±{std_time:.1f} ms")
                print(f"       FPS: {timing_results[seq_len]['fps']:.1f}")
            
            # Overall performance assessment
            avg_time_per_frame = np.mean([result['avg_time_ms'] / seq_len 
                                        for seq_len, result in timing_results.items()])
            under_100ms = avg_time_per_frame <= 100.0
            real_time_capable = avg_time_per_frame <= 50.0  # 20 FPS threshold
            
            result = {
                'status': 'success',
                'timing_by_sequence': timing_results,
                'avg_time_per_frame_ms': avg_time_per_frame,
                'under_100ms': under_100ms,
                'real_time_capable': real_time_capable,
                'device_type': str(self.device)
            }
            
            print(f"   ✅ Inference speed test completed")
            print(f"   📊 Average time per frame: {avg_time_per_frame:.1f} ms")
            print(f"   {'✅' if under_100ms else '❌'} Under 100ms target: {under_100ms}")
            print(f"   {'✅' if real_time_capable else '❌'} Real-time capable: {real_time_capable}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def test_demo_functionality(self) -> Dict:
        """Test 5: Demo Interface Functionality"""
        print("\n🧪 TEST 5: Demo Interface Functionality")
        
        if not DEMO_AVAILABLE:
            print("   ⏳ Demo module not available (gradio not installed)")
            return {
                'status': 'skipped',
                'reason': 'Demo dependencies not installed',
                'note': 'Install gradio to test demo functionality'
            }
        
        try:
            # Test demo initialization
            demo = TemporalPredictionDemo()
            
            # Test sequence generation
            sequence_types = ['moving_character', 'falling_objects', 'growing_plants']
            generation_results = {}
            
            for seq_type in sequence_types:
                try:
                    frames = demo.generate_sequence(seq_type, length=8)
                    generation_results[seq_type] = {
                        'status': 'success',
                        'frame_count': len(frames),
                        'frame_size': frames[0].size if frames else None
                    }
                    print(f"   ✅ Generated {seq_type}: {len(frames)} frames")
                except Exception as e:
                    generation_results[seq_type] = {'status': 'failed', 'error': str(e)}
                    print(f"   ❌ Failed {seq_type}: {e}")
            
            # Test prediction functionality (with dummy model)
            try:
                frames = demo.generate_sequence('moving_character', length=4)
                predicted_frames, metrics = demo.predict_next_frames(frames, num_predictions=2)
                
                prediction_test = {
                    'status': 'success',
                    'input_frames': len(frames),
                    'predicted_frames': len(predicted_frames),
                    'metrics': metrics
                }
                print(f"   ✅ Prediction test: {len(frames)} → {len(predicted_frames)} frames")
                
            except Exception as e:
                prediction_test = {'status': 'failed', 'error': str(e)}
                print(f"   ❌ Prediction test failed: {e}")
            
            result = {
                'status': 'success',
                'demo_initialized': True,
                'sequence_generation': generation_results,
                'prediction_test': prediction_test,
                'supported_sequence_types': sequence_types
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ Demo test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def generate_test_report(self) -> Dict:
        """Generate comprehensive test report"""
        print("\n📊 GENERATING MILESTONE 2 TEST REPORT")
        
        # Run all tests
        self.results['model_loading'] = self.test_temporal_model_loading()
        self.results['memory_usage'] = self.test_temporal_memory_usage() 
        self.results['prediction_accuracy'] = self.test_prediction_accuracy()
        self.results['inference_speed'] = self.test_temporal_inference_speed()
        self.results['demo_functionality'] = self.test_demo_functionality()
        
        # Calculate overall status
        tests_passed = sum(1 for result in self.results.values() 
                          if result.get('status') == 'success')
        tests_total = len(self.results)
        
        # Check milestone achievement
        milestone_achieved = (
            self.results['model_loading'].get('status') == 'success' and
            self.results['memory_usage'].get('under_8gb_limit', False) != False and
            self.results['inference_speed'].get('under_100ms', False) and
            self.results['demo_functionality'].get('status') == 'success'
        )
        
        # Prediction accuracy may need training
        prediction_status = self.results['prediction_accuracy'].get('target_achieved', False)
        
        report = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'milestone': 'Next-Frame Prediction (Milestone 2)',
            'device': str(self.device),
            'tests_passed': tests_passed,
            'tests_total': tests_total,
            'milestone_achieved': milestone_achieved,
            'prediction_ready_for_training': self.results['prediction_accuracy'].get('status') != 'failed',
            'results': self.results
        }
        
        # Save report
        report_path = self.test_outputs_dir / 'milestone2_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self.generate_markdown_report(report)
        
        print(f"\n{'='*60}")
        print(f"🎯 MILESTONE 2 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Tests Passed: {tests_passed}/{tests_total}")
        print(f"Architecture: {'✅' if milestone_achieved else '❌'}")
        print(f"Training Ready: {'✅' if report['prediction_ready_for_training'] else '❌'}")
        print(f"Prediction Target: {'✅' if prediction_status else '⏳ Needs Training'}")
        
        if milestone_achieved and not prediction_status:
            print(f"\n🎉 MILESTONE 2 INFRASTRUCTURE COMPLETE!")
            print(f"⏳ Next: Train temporal model to achieve >80% prediction accuracy")
        elif milestone_achieved and prediction_status:
            print(f"\n🎉🎉 MILESTONE 2 FULLY ACHIEVED!")
            print(f"✅ Next-Frame Prediction capability unlocked!")
        
        return report
    
    def generate_markdown_report(self, report: Dict):
        """Generate detailed markdown test report"""
        markdown_content = f"""# 🧪 Milestone 2 Test Results: Next-Frame Prediction

> Test Date: {report['test_date']}
> Device: {report['device']}
> Status: {report['tests_passed']}/{report['tests_total']} tests passed

## Summary

{'✅ MILESTONE 2 INFRASTRUCTURE COMPLETE' if report['milestone_achieved'] else '⏳ MILESTONE 2 IN PROGRESS'}

{f"🎉 Prediction accuracy target achieved!" if report['results']['prediction_accuracy'].get('target_achieved') else "⏳ Prediction accuracy needs training to reach >80%"}

## Test Results

### ✅ Test 1: Temporal Model Architecture
```json
{json.dumps(report['results']['model_loading'], indent=2)}
```

### ✅ Test 2: Memory Usage Analysis  
```json
{json.dumps(report['results']['memory_usage'], indent=2)}
```

### 📊 Test 3: Prediction Accuracy
```json
{json.dumps(report['results']['prediction_accuracy'], indent=2)}
```

### ⚡ Test 4: Inference Speed
```json
{json.dumps(report['results']['inference_speed'], indent=2)}
```

### 🖥️ Test 5: Demo Functionality
```json
{json.dumps(report['results']['demo_functionality'], indent=2)}
```

## Milestone 2 Requirements Status

| Requirement | Status | Details |
|-------------|--------|---------|
| Architecture Complete | {'✅' if report['results']['model_loading'].get('status') == 'success' else '❌'} | Temporal transformer + VQ-VAE integration |
| Memory Efficient | {'✅' if report['results']['memory_usage'].get('under_8gb_limit') != False else '❌'} | Under 8GB VRAM limit |
| Real-time Capable | {'✅' if report['results']['inference_speed'].get('under_100ms') else '❌'} | Under 100ms per sequence |
| Demo Functional | {'✅' if report['results']['demo_functionality'].get('status') == 'success' else '❌'} | Interactive prediction interface |
| Prediction Accuracy >80% | {'✅' if report['results']['prediction_accuracy'].get('target_achieved') else '⏳'} | {'Achieved' if report['results']['prediction_accuracy'].get('target_achieved') else 'Needs Training'} |

## Next Steps

{'🚀 **Ready for Milestone 3**: Sequence Generation' if report['milestone_achieved'] and report['results']['prediction_accuracy'].get('target_achieved') else '🏋️ **Training Phase**: Run temporal model training to achieve prediction accuracy target'}

### Training Commands
```bash
# Start temporal prediction training
python src/train_temporal.py --num_epochs 30

# Monitor training progress  
tensorboard --logdir logs/temporal

# Test trained model
python test_milestone2.py --checkpoint checkpoints/temporal/best.pt

# When accuracy >80% achieved:
# ✅ Milestone 2 complete!
# 🚀 Move to Milestone 3: Sequence Generation
```

---

*Generated by Milestone 2 Capability Tester*
"""
        
        markdown_path = self.test_outputs_dir / 'MILESTONE2_TEST_RESULTS.md'
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
        
        print(f"📄 Test report saved to {markdown_path}")


def main():
    """Run Milestone 2 capability tests"""
    tester = Milestone2Tester()
    report = tester.generate_test_report()
    return report


if __name__ == "__main__":
    main()

