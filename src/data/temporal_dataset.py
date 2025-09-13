"""
Temporal Dataset for Milestone 2: Next-Frame Prediction
Handles video sequences and temporal relationships for training dynamics models
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class TemporalGameDataset(Dataset):
    """Dataset for temporal game sequences with frame prediction"""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 16,
        prediction_horizon: int = 1,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        generate_synthetic: bool = True,
        frame_skip: int = 1,
        overlap_ratio: float = 0.5
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.split = split
        self.frame_skip = frame_skip
        self.overlap_ratio = overlap_ratio
        
        # Setup transforms
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform
        
        # Generate synthetic data if needed
        if generate_synthetic:
            self._generate_synthetic_sequences()
        
        # Load sequences
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} temporal sequences for {split}")
    
    def _default_transform(self):
        """Default transform for video frames"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _generate_synthetic_sequences(self):
        """Generate synthetic game sequences with temporal dynamics"""
        synthetic_dir = self.data_dir / 'synthetic_sequences'
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already generated
        if len(list(synthetic_dir.glob('*.npz'))) >= 100:
            print("Synthetic sequences already exist")
            return
        
        print("Generating synthetic temporal sequences...")
        
        # Generate different types of sequences
        sequence_types = [
            'moving_character',
            'falling_objects',
            'growing_plants',
            'water_animation',
            'particle_effects'
        ]
        
        num_sequences_per_type = 20
        
        for seq_type in sequence_types:
            for seq_id in tqdm(range(num_sequences_per_type), desc=f"Generating {seq_type}"):
                sequence = self._create_sequence_type(seq_type, length=32)
                
                # Save sequence
                filename = f"{seq_type}_{seq_id:03d}.npz"
                np.savez_compressed(
                    synthetic_dir / filename,
                    frames=sequence['frames'],
                    actions=sequence.get('actions'),
                    metadata=sequence.get('metadata', {})
                )
    
    def _create_sequence_type(self, seq_type: str, length: int = 32) -> Dict[str, Any]:
        """Create a specific type of synthetic sequence"""
        frames = []
        actions = []
        
        # Initialize scene
        scene_state = self._initialize_scene(seq_type)
        
        for frame_idx in range(length):
            # Update scene based on type
            if seq_type == 'moving_character':
                scene_state, action = self._update_moving_character(scene_state, frame_idx)
            elif seq_type == 'falling_objects':
                scene_state, action = self._update_falling_objects(scene_state, frame_idx)
            elif seq_type == 'growing_plants':
                scene_state, action = self._update_growing_plants(scene_state, frame_idx)
            elif seq_type == 'water_animation':
                scene_state, action = self._update_water_animation(scene_state, frame_idx)
            elif seq_type == 'particle_effects':
                scene_state, action = self._update_particle_effects(scene_state, frame_idx)
            else:
                scene_state, action = self._update_default(scene_state, frame_idx)
            
            # Render frame
            frame = self._render_scene(scene_state, seq_type)
            frames.append(frame)
            actions.append(action)
        
        return {
            'frames': np.array(frames),
            'actions': np.array(actions),
            'metadata': {'type': seq_type, 'length': length}
        }
    
    def _initialize_scene(self, seq_type: str) -> Dict[str, Any]:
        """Initialize scene state for different sequence types"""
        if seq_type == 'moving_character':
            return {
                'character_pos': [50, 200],
                'character_vel': [random.uniform(1, 3), 0],
                'platforms': [[0, 220, 256, 240], [100, 180, 200, 200]],
                'direction': 1
            }
        elif seq_type == 'falling_objects':
            return {
                'objects': [[random.randint(20, 236), 0, random.uniform(1, 3)] 
                           for _ in range(random.randint(2, 5))],
                'ground_y': 220
            }
        elif seq_type == 'growing_plants':
            return {
                'plants': [[50 + i * 40, 220, random.uniform(0.5, 2.0)] 
                          for i in range(5)],
                'growth_stage': 0
            }
        elif seq_type == 'water_animation':
            return {
                'wave_phase': 0,
                'wave_amplitude': 10,
                'water_level': 200
            }
        elif seq_type == 'particle_effects':
            return {
                'particles': [[128, 128, random.uniform(-2, 2), random.uniform(-3, -1), 
                              random.randint(10, 30)] for _ in range(20)],
                'emitter_pos': [128, 128]
            }
        else:
            return {}
    
    def _update_moving_character(self, state: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Update moving character sequence"""
        # Simple physics
        state['character_pos'][0] += state['character_vel'][0] * state['direction']
        
        # Bounce off walls
        if state['character_pos'][0] <= 10 or state['character_pos'][0] >= 240:
            state['direction'] *= -1
        
        # Add some vertical movement (jumping)
        if frame_idx % 20 == 0:
            state['character_vel'][1] = -4
        
        state['character_vel'][1] += 0.3  # gravity
        state['character_pos'][1] += state['character_vel'][1]
        
        # Landing on platforms or ground
        for platform in state['platforms']:
            if (platform[0] <= state['character_pos'][0] <= platform[2] and 
                platform[1] <= state['character_pos'][1] <= platform[3]):
                state['character_pos'][1] = platform[1] - 16
                state['character_vel'][1] = 0
                break
        
        # Action: 0=left, 1=right, 2=jump, 3=idle
        if state['direction'] > 0:
            action = 1  # moving right
        else:
            action = 0  # moving left
        
        if frame_idx % 20 == 0:
            action = 2  # jumping
        
        return state, action
    
    def _update_falling_objects(self, state: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Update falling objects sequence"""
        # Update existing objects
        for obj in state['objects']:
            obj[1] += obj[2]  # y += velocity
        
        # Remove objects that hit the ground
        state['objects'] = [obj for obj in state['objects'] if obj[1] < state['ground_y']]
        
        # Add new objects occasionally
        if frame_idx % 10 == 0 and len(state['objects']) < 8:
            state['objects'].append([random.randint(20, 236), 0, random.uniform(1, 3)])
        
        return state, 3  # idle action
    
    def _update_growing_plants(self, state: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Update growing plants sequence"""
        state['growth_stage'] = min(frame_idx / 10.0, 5.0)  # Grow over time
        
        return state, 3  # idle action
    
    def _update_water_animation(self, state: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Update water animation sequence"""
        state['wave_phase'] = frame_idx * 0.2
        
        return state, 3  # idle action
    
    def _update_particle_effects(self, state: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Update particle effects sequence"""
        # Update existing particles
        for particle in state['particles']:
            particle[0] += particle[2]  # x += vx
            particle[1] += particle[3]  # y += vy
            particle[4] -= 1  # life -= 1
        
        # Remove dead particles
        state['particles'] = [p for p in state['particles'] if p[4] > 0]
        
        # Add new particles
        while len(state['particles']) < 15:
            state['particles'].append([
                state['emitter_pos'][0] + random.uniform(-5, 5),
                state['emitter_pos'][1] + random.uniform(-5, 5),
                random.uniform(-2, 2),
                random.uniform(-3, -1),
                random.randint(20, 40)
            ])
        
        return state, 3  # idle action
    
    def _update_default(self, state: Dict, frame_idx: int) -> Tuple[Dict, int]:
        """Default update for unspecified sequence types"""
        return state, 3
    
    def _render_scene(self, state: Dict, seq_type: str) -> np.ndarray:
        """Render scene state to a frame"""
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(128):
            color = int(135 + y * 0.5)
            frame[y, :] = [color, color + 20, 255]
        
        # Ground
        frame[220:, :] = [34, 139, 34]
        
        if seq_type == 'moving_character':
            # Draw platforms
            for platform in state['platforms']:
                frame[platform[1]:platform[3], platform[0]:platform[2]] = [139, 69, 19]
            
            # Draw character
            char_x = int(state['character_pos'][0])
            char_y = int(state['character_pos'][1])
            if 0 <= char_x < 240 and 0 <= char_y < 240:
                frame[char_y:char_y+16, char_x:char_x+16] = [255, 0, 0]
        
        elif seq_type == 'falling_objects':
            for obj in state['objects']:
                x, y = int(obj[0]), int(obj[1])
                if 0 <= x < 248 and 0 <= y < 248:
                    frame[y:y+8, x:x+8] = [255, 215, 0]  # Gold objects
        
        elif seq_type == 'growing_plants':
            for plant in state['plants']:
                x, base_y, growth_rate = plant
                height = int(state['growth_stage'] * growth_rate * 10)
                height = min(height, 50)
                if height > 0:
                    frame[base_y-height:base_y, x:x+5] = [0, 255, 0]  # Green plants
        
        elif seq_type == 'water_animation':
            water_level = state['water_level']
            for x in range(256):
                wave_height = int(state['wave_amplitude'] * np.sin(state['wave_phase'] + x * 0.1))
                water_y = water_level + wave_height
                if water_y < 256:
                    frame[water_y:, x] = [0, 100, 255]  # Blue water
        
        elif seq_type == 'particle_effects':
            for particle in state['particles']:
                x, y = int(particle[0]), int(particle[1])
                if 0 <= x < 254 and 0 <= y < 254:
                    intensity = min(255, particle[4] * 8)
                    frame[y:y+2, x:x+2] = [intensity, intensity//2, 0]  # Fire-like particles
        
        return frame
    
    def _load_sequences(self) -> List[Dict[str, Any]]:
        """Load all sequence data"""
        sequences = []
        
        # Load synthetic sequences
        synthetic_dir = self.data_dir / 'synthetic_sequences'
        if synthetic_dir.exists():
            for seq_file in synthetic_dir.glob('*.npz'):
                data = np.load(seq_file, allow_pickle=True)
                
                frames = data['frames']
                actions = data.get('actions', None)
                
                # Create overlapping windows
                step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
                
                for start_idx in range(0, len(frames) - self.sequence_length - self.prediction_horizon + 1, step_size):
                    end_idx = start_idx + self.sequence_length
                    pred_end_idx = end_idx + self.prediction_horizon
                    
                    sequence_data = {
                        'input_frames': frames[start_idx:end_idx:self.frame_skip],
                        'target_frames': frames[end_idx:pred_end_idx:self.frame_skip],
                        'file': str(seq_file),
                        'start_idx': start_idx
                    }
                    
                    if actions is not None:
                        sequence_data['actions'] = actions[start_idx:end_idx:self.frame_skip]
                    
                    sequences.append(sequence_data)
        
        # Split data
        random.shuffle(sequences)
        n = len(sequences)
        
        if self.split == 'train':
            sequences = sequences[:int(0.8 * n)]
        elif self.split == 'val':
            sequences = sequences[int(0.8 * n):int(0.9 * n)]
        else:  # test
            sequences = sequences[int(0.9 * n):]
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a temporal sequence sample"""
        seq_data = self.sequences[idx]
        
        # Load input frames
        input_frames = seq_data['input_frames']
        target_frames = seq_data['target_frames']
        
        # Convert to PIL and transform
        input_tensors = []
        for frame in input_frames:
            if frame.dtype == np.uint8:
                pil_frame = Image.fromarray(frame)
            else:
                pil_frame = Image.fromarray((frame * 255).astype(np.uint8))
            
            tensor_frame = self.transform(pil_frame)
            input_tensors.append(tensor_frame)
        
        target_tensors = []
        for frame in target_frames:
            if frame.dtype == np.uint8:
                pil_frame = Image.fromarray(frame)
            else:
                pil_frame = Image.fromarray((frame * 255).astype(np.uint8))
            
            tensor_frame = self.transform(pil_frame)
            target_tensors.append(tensor_frame)
        
        result = {
            'input_frames': torch.stack(input_tensors),
            'target_frames': torch.stack(target_tensors),
            'file': seq_data['file'],
            'start_idx': seq_data['start_idx']
        }
        
        # Add actions if available
        if 'actions' in seq_data:
            result['actions'] = torch.tensor(seq_data['actions'], dtype=torch.long)
        
        return result


def create_temporal_dataloaders(
    data_dir: str,
    sequence_length: int = 16,
    prediction_horizon: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    frame_skip: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create temporal dataloaders for training"""
    
    # Create datasets
    train_dataset = TemporalGameDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        split='train',
        frame_skip=frame_skip
    )
    
    val_dataset = TemporalGameDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        split='val',
        frame_skip=frame_skip
    )
    
    test_dataset = TemporalGameDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        split='test',
        frame_skip=frame_skip
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def visualize_sequence(dataset: TemporalGameDataset, idx: int = 0):
    """Visualize a temporal sequence"""
    sample = dataset[idx]
    
    input_frames = sample['input_frames']
    target_frames = sample['target_frames']
    
    # Denormalize
    denorm = lambda x: (x * 0.5 + 0.5).clamp(0, 1)
    
    input_frames = denorm(input_frames)
    target_frames = denorm(target_frames)
    
    # Create visualization
    seq_len = len(input_frames)
    target_len = len(target_frames)
    total_frames = seq_len + target_len
    
    fig, axes = plt.subplots(2, max(seq_len, target_len), figsize=(total_frames * 2, 4))
    
    # Plot input sequence
    for i in range(seq_len):
        ax = axes[0, i] if seq_len > 1 else axes[0]
        frame = input_frames[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(frame)
        ax.set_title(f'Input {i+1}')
        ax.axis('off')
    
    # Plot target sequence
    for i in range(target_len):
        if target_len > 1:
            ax = axes[1, i]
        else:
            ax = axes[1]
        frame = target_frames[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(frame)
        ax.set_title(f'Target {i+1}')
        ax.axis('off')
    
    plt.suptitle('Temporal Sequence: Input → Target')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test the temporal dataset
    print("Testing TemporalGameDataset...")
    
    # Create dataset
    dataset = TemporalGameDataset(
        data_dir="datasets/temporal",
        sequence_length=8,
        prediction_horizon=2,
        generate_synthetic=True
    )
    
    # Test sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input frames shape: {sample['input_frames'].shape}")
    print(f"Target frames shape: {sample['target_frames'].shape}")
    
    # Test dataloader
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        data_dir="datasets/temporal",
        sequence_length=8,
        batch_size=4
    )
    
    # Get batch
    batch = next(iter(train_loader))
    print(f"Batch input shape: {batch['input_frames'].shape}")
    print(f"Batch target shape: {batch['target_frames'].shape}")
    
    print("✅ Temporal dataset test successful!")
    
    # Visualize if requested
    if '--viz' in os.sys.argv:
        visualize_sequence(dataset, 0)