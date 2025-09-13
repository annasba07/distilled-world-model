"""
Dataset and DataLoader for game images
Supports multiple sources: local files, procedural generation, and downloaded samples
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import requests
import zipfile
import json
from tqdm import tqdm


class GameImageDataset(Dataset):
    """Dataset for game images - supports static frames and sequences"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (256, 256),
        sequence_length: int = 1,  # 1 for static, >1 for sequences
        download: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.sequence_length = sequence_length
        
        # Setup transforms
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform
        
        # Download sample data if needed
        if download and not self._check_data_exists():
            self._download_sample_data()
        
        # Load file paths
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _default_transform(self):
        """Default transform pipeline"""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _check_data_exists(self) -> bool:
        """Check if data directory exists and has content"""
        return self.data_dir.exists() and any(self.data_dir.iterdir())
    
    def _download_sample_data(self):
        """Download sample game sprites/screenshots"""
        print("Downloading sample game data...")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate procedural samples for testing
        self._generate_procedural_samples(100)
        
        # Download open game art (simplified for demo)
        # In production, use the full data collection pipeline
        print("Sample data generated successfully!")
    
    def _generate_procedural_samples(self, num_samples: int):
        """Generate simple procedural game-like images"""
        samples_dir = self.data_dir / 'procedural'
        samples_dir.mkdir(exist_ok=True)
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Create simple game-like scene
            img = self._create_game_scene()
            
            # Save image
            img_path = samples_dir / f"sample_{i:04d}.png"
            img.save(img_path)
    
    def _create_game_scene(self) -> Image.Image:
        """Create a simple procedural game scene"""
        # Create blank canvas
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(128):
            color = int(135 + y * 0.5)  # Light blue gradient
            img_array[y, :] = [color, color + 20, 255]
        
        # Ground
        img_array[128:, :] = [34, 139, 34]  # Green ground
        
        # Add random platforms
        num_platforms = random.randint(2, 5)
        for _ in range(num_platforms):
            x = random.randint(0, 200)
            y = random.randint(140, 220)
            w = random.randint(30, 80)
            h = 10
            img_array[y:y+h, x:x+w] = [139, 69, 19]  # Brown platforms
        
        # Add character (simple square)
        char_x = random.randint(10, 240)
        char_y = random.randint(100, 240)
        img_array[char_y:char_y+16, char_x:char_x+16] = [255, 0, 0]  # Red character
        
        # Add collectibles
        num_coins = random.randint(1, 3)
        for _ in range(num_coins):
            coin_x = random.randint(5, 250)
            coin_y = random.randint(50, 240)
            img_array[coin_y:coin_y+8, coin_x:coin_x+8] = [255, 215, 0]  # Gold coins
        
        return Image.fromarray(img_array)
    
    def _load_samples(self) -> List[Path]:
        """Load all image paths"""
        samples = []
        
        # Supported image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        
        # Find all images
        for ext in extensions:
            samples.extend(self.data_dir.rglob(ext))
        
        # Split data (80/10/10)
        random.shuffle(samples)
        n = len(samples)
        
        if self.split == 'train':
            samples = samples[:int(0.8 * n)]
        elif self.split == 'val':
            samples = samples[int(0.8 * n):int(0.9 * n)]
        else:  # test
            samples = samples[int(0.9 * n):]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample or sequence"""
        
        if self.sequence_length == 1:
            # Single image
            img_path = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return {
                'images': img,
                'paths': str(img_path)
            }
        else:
            # Image sequence
            # For now, create synthetic sequence by augmenting single image
            # In production, load actual video sequences
            base_idx = idx
            images = []
            
            for i in range(self.sequence_length):
                # Cycle through available images
                img_idx = (base_idx + i) % len(self.samples)
                img_path = self.samples[img_idx]
                img = Image.open(img_path).convert('RGB')
                
                # Apply slight augmentation for temporal variation
                if i > 0:
                    img = transforms.functional.adjust_brightness(img, 1.0 + i * 0.02)
                
                if self.transform:
                    img = self.transform(img)
                
                images.append(img)
            
            return {
                'images': torch.stack(images),
                'paths': str(self.samples[base_idx])
            }


class GameVideoDataset(Dataset):
    """Dataset for game video sequences"""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 32,
        stride: int = 1,
        transform: Optional[transforms.Compose] = None,
        split: str = 'train'
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # Load sequences
        self.sequences = self._load_sequences()
    
    def _load_sequences(self) -> List[Dict]:
        """Load video sequences from disk"""
        sequences = []
        
        # Look for .npz files containing sequences
        sequence_files = list(self.data_dir.glob('*.npz'))
        
        for seq_file in sequence_files:
            data = np.load(seq_file)
            if 'frames' in data:
                frames = data['frames']
                actions = data.get('actions', None)
                
                # Create sliding windows
                for start_idx in range(0, len(frames) - self.sequence_length, self.stride):
                    end_idx = start_idx + self.sequence_length
                    sequences.append({
                        'frames': frames[start_idx:end_idx],
                        'actions': actions[start_idx:end_idx] if actions is not None else None,
                        'file': str(seq_file),
                        'start_idx': start_idx
                    })
        
        # Split sequences
        random.shuffle(sequences)
        n = len(sequences)
        
        if self.split == 'train':
            sequences = sequences[:int(0.8 * n)]
        elif self.split == 'val':
            sequences = sequences[int(0.8 * n):int(0.9 * n)]
        else:
            sequences = sequences[int(0.9 * n):]
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_data = self.sequences[idx]
        
        # Load frames
        frames = seq_data['frames']
        
        # Convert to PIL and transform
        transformed_frames = []
        for frame in frames:
            if frame.dtype == np.uint8:
                pil_frame = Image.fromarray(frame)
            else:
                # Assume float in [0, 1]
                pil_frame = Image.fromarray((frame * 255).astype(np.uint8))
            
            if self.transform:
                transformed_frames.append(self.transform(pil_frame))
            else:
                transformed_frames.append(
                    transforms.ToTensor()(pil_frame)
                )
        
        output = {
            'frames': torch.stack(transformed_frames),
            'file': seq_data['file'],
            'start_idx': seq_data['start_idx']
        }
        
        if seq_data['actions'] is not None:
            output['actions'] = torch.tensor(seq_data['actions'], dtype=torch.long)
        
        return output


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    sequence_length: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders"""
    
    # Create datasets
    train_dataset = GameImageDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        sequence_length=sequence_length
    )
    
    val_dataset = GameImageDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        sequence_length=sequence_length
    )
    
    test_dataset = GameImageDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        sequence_length=sequence_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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


if __name__ == "__main__":
    # Test the dataset
    print("Testing GameImageDataset...")
    
    # Create dataset
    dataset = GameImageDataset(
        data_dir="datasets/raw",
        split='train',
        download=True
    )
    
    # Test single sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['images'].shape}")
    print(f"Image range: [{sample['images'].min():.2f}, {sample['images'].max():.2f}]")
    
    # Test dataloader
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="datasets/raw",
        batch_size=4
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch images shape: {batch['images'].shape}")
    
    print("✅ Dataset test successful!")