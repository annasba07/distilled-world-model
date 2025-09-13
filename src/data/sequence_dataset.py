"""
Sequence Dataset for Milestone 3: Short Sequence Generation
Extended dataset for training 30-frame coherent sequences
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional
import random
import math
import cv2


class LongSequenceDataset(Dataset):
    """Dataset for training long sequence generation (Milestone 3)"""
    
    def __init__(
        self,
        sequence_length: int = 32,
        num_sequences: int = 1000,
        image_size: Tuple[int, int] = (256, 256),
        num_objects_range: Tuple[int, int] = (1, 4),
        physics_enabled: bool = True,
        diversity_level: float = 1.0
    ):
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.image_size = image_size
        self.num_objects_range = num_objects_range
        self.physics_enabled = physics_enabled
        self.diversity_level = diversity_level
        
        # Scene types for diverse generation
        self.scene_types = [
            'platformer_physics',
            'top_down_movement', 
            'particle_systems',
            'growing_structures',
            'bouncing_balls',
            'falling_leaves',
            'spinning_objects',
            'morphing_shapes'
        ]
        
        # Color palettes for visual diversity
        self.color_palettes = [
            [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)],  # Bright
            [(200, 150, 100), (150, 200, 150), (100, 150, 200), (200, 200, 100)],  # Muted
            [(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 200)],  # Pastel
            [(150, 50, 50), (50, 150, 50), (50, 50, 150), (150, 150, 50)]          # Dark
        ]
        
        print(f"[INIT] LongSequenceDataset initialized:")
        print(f"   Sequences: {num_sequences}")
        print(f"   Length: {sequence_length} frames")
        print(f"   Size: {image_size}")
        print(f"   Scene types: {len(self.scene_types)}")
    
    def _safe_randint(self, low: int, high: int) -> int:
        """Safe randint that handles invalid ranges"""
        if low >= high:
            return low
        return random.randint(low, high)
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Generate a long sequence with coherent motion and physics"""
        
        # Select scene type and parameters
        scene_type = random.choice(self.scene_types)
        color_palette = random.choice(self.color_palettes)
        
        # Generate sequence based on scene type
        if scene_type == 'platformer_physics':
            frames, metadata = self._generate_platformer_sequence(color_palette)
        elif scene_type == 'top_down_movement':
            frames, metadata = self._generate_topdown_sequence(color_palette)
        elif scene_type == 'particle_systems':
            frames, metadata = self._generate_particle_sequence(color_palette)
        elif scene_type == 'growing_structures':
            frames, metadata = self._generate_growing_sequence(color_palette)
        elif scene_type == 'bouncing_balls':
            frames, metadata = self._generate_bouncing_sequence(color_palette)
        elif scene_type == 'falling_leaves':
            frames, metadata = self._generate_falling_sequence(color_palette)
        elif scene_type == 'spinning_objects':
            frames, metadata = self._generate_spinning_sequence(color_palette)
        else:  # morphing_shapes
            frames, metadata = self._generate_morphing_sequence(color_palette)
        
        # Convert to tensor
        frame_tensor = torch.stack([
            torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            for frame in frames
        ])
        
        metadata.update({
            'scene_type': scene_type,
            'sequence_id': idx,
            'length': len(frames)
        })
        
        return frame_tensor, metadata
    
    def _generate_platformer_sequence(self, colors: List[Tuple]) -> Tuple[List[Image.Image], Dict]:
        """Generate platformer-style sequence with gravity and collisions"""
        frames = []
        width, height = self.image_size
        
        # Initialize character and platforms
        char_x = width // 4
        char_y = height // 2
        char_vel_x = random.uniform(-2, 2) * self.diversity_level
        char_vel_y = 0
        char_size = random.randint(15, 25)
        
        # Create platforms
        platforms = []
        for _ in range(random.randint(2, 4)):
            # Adjust platform bounds for smaller images
            min_x = min(50, width // 4)
            max_x = max(min_x + 20, width - max(100, width // 4))
            min_y = height // 2
            max_y = max(min_y + 10, height - max(50, height // 8))
            platform_width = random.randint(min(80, width//3), min(150, width//2))
            
            platforms.append({
                'x': random.randint(min_x, max_x) if max_x > min_x else min_x,
                'y': random.randint(min_y, max_y) if max_y > min_y else min_y,
                'width': platform_width,
                'height': 20
            })
        
        gravity = 0.3
        bounce_damping = 0.7
        ground_y = height - 30
        
        for frame_idx in range(self.sequence_length):
            # Create frame
            img = Image.new('RGB', self.image_size, (50, 50, 50))
            draw = ImageDraw.Draw(img)
            
            # Draw platforms
            for platform in platforms:
                draw.rectangle([
                    platform['x'], platform['y'],
                    platform['x'] + platform['width'], platform['y'] + platform['height']
                ], fill=colors[1])
            
            # Update character physics
            if self.physics_enabled:
                # Apply gravity
                char_vel_y += gravity
                
                # Update position
                char_x += char_vel_x
                char_y += char_vel_y
                
                # Ground collision
                if char_y + char_size > ground_y:
                    char_y = ground_y - char_size
                    char_vel_y = -char_vel_y * bounce_damping
                
                # Platform collisions
                for platform in platforms:
                    if (char_x + char_size > platform['x'] and 
                        char_x < platform['x'] + platform['width'] and
                        char_y + char_size > platform['y'] and
                        char_y < platform['y'] + platform['height']):
                        
                        if char_vel_y > 0:  # Falling onto platform
                            char_y = platform['y'] - char_size
                            char_vel_y = -char_vel_y * bounce_damping
                
                # Wall bouncing
                if char_x <= 0 or char_x + char_size >= width:
                    char_vel_x = -char_vel_x * bounce_damping
                    char_x = max(0, min(width - char_size, char_x))
            
            # Draw character
            draw.ellipse([
                char_x, char_y,
                char_x + char_size, char_y + char_size
            ], fill=colors[0])
            
            # Add motion trails for visual appeal
            if frame_idx > 0:
                trail_alpha = max(0.3, 1.0 - frame_idx * 0.1)
                trail_size = int(char_size * trail_alpha)
                if trail_size > 2:
                    draw.ellipse([
                        char_x - char_vel_x + char_size//2 - trail_size//2,
                        char_y - char_vel_y + char_size//2 - trail_size//2,
                        char_x - char_vel_x + char_size//2 + trail_size//2,
                        char_y - char_vel_y + char_size//2 + trail_size//2
                    ], fill=tuple(int(c * trail_alpha) for c in colors[0]))
            
            frames.append(img)
        
        metadata = {
            'type': 'platformer_physics',
            'gravity': gravity,
            'num_platforms': len(platforms),
            'character_size': char_size
        }
        
        return frames, metadata
    
    def _generate_topdown_sequence(self, colors: List[Tuple]) -> Tuple[List[Image.Image], Dict]:
        """Generate top-down movement sequence"""
        frames = []
        width, height = self.image_size
        
        # Multiple moving objects
        objects = []
        for i in range(random.randint(*self.num_objects_range)):
            objects.append({
                'x': random.uniform(50, width-50),
                'y': random.uniform(50, height-50),
                'vel_x': random.uniform(-3, 3) * self.diversity_level,
                'vel_y': random.uniform(-3, 3) * self.diversity_level,
                'size': random.randint(10, 30),
                'color': colors[i % len(colors)]
            })
        
        for frame_idx in range(self.sequence_length):
            # Create frame
            img = Image.new('RGB', self.image_size, (30, 30, 40))
            draw = ImageDraw.Draw(img)
            
            # Update and draw objects
            for obj in objects:
                # Update position
                obj['x'] += obj['vel_x']
                obj['y'] += obj['vel_y']
                
                # Boundary bouncing
                if obj['x'] <= obj['size'] or obj['x'] >= width - obj['size']:
                    obj['vel_x'] = -obj['vel_x']
                    obj['x'] = max(obj['size'], min(width - obj['size'], obj['x']))
                
                if obj['y'] <= obj['size'] or obj['y'] >= height - obj['size']:
                    obj['vel_y'] = -obj['vel_y']
                    obj['y'] = max(obj['size'], min(height - obj['size'], obj['y']))
                
                # Draw object with shadow
                shadow_offset = 3
                draw.ellipse([
                    obj['x'] + shadow_offset - obj['size'],
                    obj['y'] + shadow_offset - obj['size'],
                    obj['x'] + shadow_offset + obj['size'],
                    obj['y'] + shadow_offset + obj['size']
                ], fill=(20, 20, 20))
                
                draw.ellipse([
                    obj['x'] - obj['size'], obj['y'] - obj['size'],
                    obj['x'] + obj['size'], obj['y'] + obj['size']
                ], fill=obj['color'])
            
            frames.append(img)
        
        metadata = {
            'type': 'top_down_movement',
            'num_objects': len(objects),
            'avg_speed': np.mean([abs(obj['vel_x']) + abs(obj['vel_y']) for obj in objects])
        }
        
        return frames, metadata
    
    def _generate_particle_sequence(self, colors: List[Tuple]) -> Tuple[List[Image.Image], Dict]:
        """Generate particle system sequence"""
        frames = []
        width, height = self.image_size
        
        # Particle system parameters
        num_particles = random.randint(20, 50)
        emitter_x = width // 2
        emitter_y = height // 2
        
        particles = []
        
        for frame_idx in range(self.sequence_length):
            # Create frame
            img = Image.new('RGB', self.image_size, (10, 10, 20))
            draw = ImageDraw.Draw(img)
            
            # Emit new particles
            if len(particles) < num_particles and frame_idx < self.sequence_length * 0.7:
                for _ in range(random.randint(1, 3)):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(1, 4) * self.diversity_level
                    particles.append({
                        'x': emitter_x + random.uniform(-10, 10),
                        'y': emitter_y + random.uniform(-10, 10),
                        'vel_x': math.cos(angle) * speed,
                        'vel_y': math.sin(angle) * speed,
                        'life': random.uniform(10, 30),
                        'max_life': random.uniform(10, 30),
                        'size': random.uniform(2, 8),
                        'color': random.choice(colors)
                    })
            
            # Update and draw particles
            active_particles = []
            for particle in particles:
                # Update position and life
                particle['x'] += particle['vel_x']
                particle['y'] += particle['vel_y']
                particle['life'] -= 1
                
                # Apply some drag
                particle['vel_x'] *= 0.98
                particle['vel_y'] *= 0.98
                
                # Keep particle if still alive and in bounds
                if (particle['life'] > 0 and 
                    0 <= particle['x'] <= width and 
                    0 <= particle['y'] <= height):
                    
                    # Calculate alpha based on remaining life
                    alpha = particle['life'] / particle['max_life']
                    current_size = particle['size'] * alpha
                    
                    if current_size > 1:
                        # Fade color
                        faded_color = tuple(int(c * alpha) for c in particle['color'])
                        
                        draw.ellipse([
                            particle['x'] - current_size, particle['y'] - current_size,
                            particle['x'] + current_size, particle['y'] + current_size
                        ], fill=faded_color)
                    
                    active_particles.append(particle)
            
            particles = active_particles
            
            # Draw emitter
            draw.ellipse([
                emitter_x - 5, emitter_y - 5,
                emitter_x + 5, emitter_y + 5
            ], fill=(255, 255, 255))
            
            frames.append(img)
        
        metadata = {
            'type': 'particle_system',
            'max_particles': num_particles,
            'emitter_pos': (emitter_x, emitter_y)
        }
        
        return frames, metadata
    
    def _generate_growing_sequence(self, colors: List[Tuple]) -> Tuple[List[Image.Image], Dict]:
        """Generate growing/evolving structures"""
        frames = []
        width, height = self.image_size
        
        # Growth parameters
        growth_centers = []
        for _ in range(random.randint(2, 4)):
            # Safe bounds for smaller images
            min_x, max_x = width//4, max(width//4 + 10, 3*width//4)
            min_y, max_y = height//4, max(height//4 + 10, 3*height//4)
            max_size = min(80, max(30, width//4))  # Scale max size to image
            
            growth_centers.append({
                'x': self._safe_randint(min_x, max_x),
                'y': self._safe_randint(min_y, max_y),
                'size': 5,
                'growth_rate': random.uniform(0.5, 2.0) * self.diversity_level,
                'max_size': self._safe_randint(30, max_size),
                'color': random.choice(colors),
                'branches': []
            })
        
        for frame_idx in range(self.sequence_length):
            # Create frame
            img = Image.new('RGB', self.image_size, (20, 25, 30))
            draw = ImageDraw.Draw(img)
            
            # Update growth centers
            for center in growth_centers:
                # Grow main structure
                if center['size'] < center['max_size']:
                    center['size'] += center['growth_rate']
                
                # Add branches randomly
                if random.random() < 0.1 and len(center['branches']) < 5:
                    angle = random.uniform(0, 2 * math.pi)
                    branch_length = random.uniform(10, center['size'])
                    center['branches'].append({
                        'angle': angle,
                        'length': branch_length,
                        'growth': 0
                    })
                
                # Grow branches
                for branch in center['branches']:
                    if branch['growth'] < branch['length']:
                        branch['growth'] += center['growth_rate'] * 0.5
                
                # Draw main structure
                draw.ellipse([
                    center['x'] - center['size'], center['y'] - center['size'],
                    center['x'] + center['size'], center['y'] + center['size']
                ], fill=center['color'])
                
                # Draw branches
                for branch in center['branches']:
                    end_x = center['x'] + math.cos(branch['angle']) * branch['growth']
                    end_y = center['y'] + math.sin(branch['angle']) * branch['growth']
                    
                    draw.line([
                        (center['x'], center['y']),
                        (end_x, end_y)
                    ], fill=center['color'], width=3)
                    
                    # Branch endpoint
                    if branch['growth'] > 5:
                        endpoint_size = min(5, branch['growth'] / 3)
                        draw.ellipse([
                            end_x - endpoint_size, end_y - endpoint_size,
                            end_x + endpoint_size, end_y + endpoint_size
                        ], fill=center['color'])
            
            frames.append(img)
        
        metadata = {
            'type': 'growing_structures',
            'num_centers': len(growth_centers),
            'total_branches': sum(len(c['branches']) for c in growth_centers)
        }
        
        return frames, metadata
    
    def _generate_bouncing_sequence(self, colors: List[Tuple]) -> Tuple[List[Image.Image], Dict]:
        """Generate bouncing balls with realistic physics"""
        frames = []
        width, height = self.image_size
        
        # Create bouncing balls  
        balls = []
        for _ in range(random.randint(2, 5)):
            # Safe bounds for different image sizes
            margin = min(50, width//6)
            min_x, max_x = margin, max(margin + 10, width - margin)
            min_y, max_y = margin, max(margin + 10, height//2)
            ball_size = self._safe_randint(8, min(20, width//15))
            
            balls.append({
                'x': random.uniform(min_x, max_x) if max_x > min_x else min_x,
                'y': random.uniform(min_y, max_y) if max_y > min_y else min_y,
                'vel_x': random.uniform(-4, 4) * self.diversity_level,
                'vel_y': random.uniform(-2, 2) * self.diversity_level,
                'size': ball_size,
                'color': random.choice(colors),
                'bounce_factor': random.uniform(0.7, 0.9)
            })
        
        gravity = 0.2
        
        for frame_idx in range(self.sequence_length):
            # Create frame
            img = Image.new('RGB', self.image_size, (40, 40, 50))
            draw = ImageDraw.Draw(img)
            
            # Update balls
            for ball in balls:
                # Apply physics
                ball['vel_y'] += gravity
                ball['x'] += ball['vel_x']
                ball['y'] += ball['vel_y']
                
                # Floor bounce
                if ball['y'] + ball['size'] > height - 10:
                    ball['y'] = height - 10 - ball['size']
                    ball['vel_y'] = -ball['vel_y'] * ball['bounce_factor']
                    ball['vel_x'] *= 0.95  # Friction
                
                # Wall bounces
                if ball['x'] <= ball['size'] or ball['x'] >= width - ball['size']:
                    ball['vel_x'] = -ball['vel_x'] * ball['bounce_factor']
                    ball['x'] = max(ball['size'], min(width - ball['size'], ball['x']))
                
                # Ceiling bounce
                if ball['y'] <= ball['size']:
                    ball['y'] = ball['size']
                    ball['vel_y'] = -ball['vel_y'] * ball['bounce_factor']
                
                # Draw ball with 3D effect
                # Shadow
                draw.ellipse([
                    ball['x'] - ball['size'] + 2, 
                    height - 8 - ball['size']//2,
                    ball['x'] + ball['size'] + 2, 
                    height - 8 + ball['size']//2
                ], fill=(20, 20, 20, 100))
                
                # Highlight
                highlight_size = ball['size'] - 3
                draw.ellipse([
                    ball['x'] - highlight_size, ball['y'] - highlight_size,
                    ball['x'] + highlight_size, ball['y'] + highlight_size
                ], fill=ball['color'])
                
                # Bright highlight
                bright_color = tuple(min(255, c + 50) for c in ball['color'])
                highlight_offset = ball['size'] // 3
                draw.ellipse([
                    ball['x'] - highlight_offset, ball['y'] - highlight_offset,
                    ball['x'], ball['y']
                ], fill=bright_color)
            
            frames.append(img)
        
        metadata = {
            'type': 'bouncing_balls',
            'num_balls': len(balls),
            'gravity': gravity
        }
        
        return frames, metadata
    
    def _generate_falling_sequence(self, colors: List[Tuple]) -> Tuple[List[Image.Image], Dict]:
        """Generate falling leaves/objects"""
        frames = []
        width, height = self.image_size
        
        leaves = []
        spawn_rate = 0.3 * self.diversity_level
        
        for frame_idx in range(self.sequence_length):
            # Create frame
            img = Image.new('RGB', self.image_size, (60, 80, 100))
            draw = ImageDraw.Draw(img)
            
            # Spawn new leaves
            if random.random() < spawn_rate:
                leaves.append({
                    'x': random.uniform(0, width),
                    'y': -10,
                    'vel_x': random.uniform(-1, 1),
                    'vel_y': random.uniform(1, 3),
                    'rotation': random.uniform(0, 360),
                    'rot_speed': random.uniform(-5, 5),
                    'size': random.randint(8, 15),
                    'color': random.choice(colors),
                    'sway_amplitude': random.uniform(0.5, 2.0),
                    'sway_frequency': random.uniform(0.1, 0.3)
                })
            
            # Update leaves
            active_leaves = []
            for leaf in leaves:
                # Update position with swaying motion
                sway_x = math.sin(frame_idx * leaf['sway_frequency']) * leaf['sway_amplitude']
                leaf['x'] += leaf['vel_x'] + sway_x
                leaf['y'] += leaf['vel_y']
                leaf['rotation'] += leaf['rot_speed']
                
                # Keep if still visible
                if leaf['y'] < height + 20:
                    # Draw leaf (simplified as rotated ellipse)
                    # Create points for rotated ellipse
                    angle = math.radians(leaf['rotation'])
                    cos_a, sin_a = math.cos(angle), math.sin(angle)
                    
                    # Ellipse points
                    w, h = leaf['size'], leaf['size'] * 0.6
                    points = []
                    for dx, dy in [(-w, -h), (w, -h), (w, h), (-w, h)]:
                        rx = dx * cos_a - dy * sin_a + leaf['x']
                        ry = dx * sin_a + dy * cos_a + leaf['y']
                        points.append((rx, ry))
                    
                    if len(points) >= 3:
                        draw.polygon(points, fill=leaf['color'])
                    
                    active_leaves.append(leaf)
            
            leaves = active_leaves
            frames.append(img)
        
        metadata = {
            'type': 'falling_leaves',
            'spawn_rate': spawn_rate,
            'max_leaves': len(leaves)
        }
        
        return frames, metadata
    
    def _generate_spinning_sequence(self, colors: List[Tuple]) -> Tuple[List[Image.Image], Dict]:
        """Generate spinning objects"""
        frames = []
        width, height = self.image_size
        
        spinners = []
        for _ in range(random.randint(2, 4)):
            spinners.append({
                'x': random.uniform(width*0.2, width*0.8),
                'y': random.uniform(height*0.2, height*0.8),
                'rotation': 0,
                'spin_speed': random.uniform(-8, 8) * self.diversity_level,
                'size': random.randint(20, 40),
                'color': random.choice(colors),
                'shape': random.choice(['square', 'triangle', 'star'])
            })
        
        for frame_idx in range(self.sequence_length):
            # Create frame
            img = Image.new('RGB', self.image_size, (30, 30, 30))
            draw = ImageDraw.Draw(img)
            
            for spinner in spinners:
                # Update rotation
                spinner['rotation'] += spinner['spin_speed']
                
                # Draw shape based on type
                angle = math.radians(spinner['rotation'])
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                
                if spinner['shape'] == 'square':
                    # Rotated square
                    size = spinner['size']
                    points = []
                    for dx, dy in [(-size, -size), (size, -size), (size, size), (-size, size)]:
                        rx = dx * cos_a - dy * sin_a + spinner['x']
                        ry = dx * sin_a + dy * cos_a + spinner['y']
                        points.append((rx, ry))
                    draw.polygon(points, fill=spinner['color'])
                    
                elif spinner['shape'] == 'triangle':
                    # Rotated triangle
                    size = spinner['size']
                    points = []
                    for dx, dy in [(0, -size), (size*0.866, size*0.5), (-size*0.866, size*0.5)]:
                        rx = dx * cos_a - dy * sin_a + spinner['x']
                        ry = dx * sin_a + dy * cos_a + spinner['y']
                        points.append((rx, ry))
                    draw.polygon(points, fill=spinner['color'])
                    
                else:  # star
                    # Rotated star (simplified)
                    size = spinner['size']
                    points = []
                    for i in range(10):  # 5-pointed star
                        angle_offset = i * math.pi / 5
                        radius = size if i % 2 == 0 else size * 0.4
                        dx = radius * math.cos(angle_offset)
                        dy = radius * math.sin(angle_offset)
                        rx = dx * cos_a - dy * sin_a + spinner['x']
                        ry = dx * sin_a + dy * cos_a + spinner['y']
                        points.append((rx, ry))
                    draw.polygon(points, fill=spinner['color'])
            
            frames.append(img)
        
        metadata = {
            'type': 'spinning_objects',
            'num_spinners': len(spinners),
            'shapes': [s['shape'] for s in spinners]
        }
        
        return frames, metadata
    
    def _generate_morphing_sequence(self, colors: List[Tuple]) -> Tuple[List[Image.Image], Dict]:
        """Generate morphing shapes"""
        frames = []
        width, height = self.image_size
        
        # Morphing parameters
        center_x, center_y = width // 2, height // 2
        base_size = random.randint(30, 60)
        num_vertices = random.randint(6, 12)
        morph_speed = random.uniform(0.1, 0.3) * self.diversity_level
        
        for frame_idx in range(self.sequence_length):
            # Create frame
            img = Image.new('RGB', self.image_size, (25, 25, 35))
            draw = ImageDraw.Draw(img)
            
            # Create morphing shape
            points = []
            for i in range(num_vertices):
                angle = (2 * math.pi * i / num_vertices) + frame_idx * 0.1
                
                # Vary radius with sin waves for morphing effect
                radius_variation = math.sin(frame_idx * morph_speed + i) * 0.3
                radius = base_size * (1 + radius_variation)
                
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))
            
            # Draw morphing shape with gradient effect
            if len(points) >= 3:
                # Main shape
                color_idx = int(frame_idx / (self.sequence_length / len(colors))) % len(colors)
                draw.polygon(points, fill=colors[color_idx])
                
                # Inner shape (smaller, different color)
                inner_points = []
                for x, y in points:
                    inner_x = center_x + (x - center_x) * 0.6
                    inner_y = center_y + (y - center_y) * 0.6
                    inner_points.append((inner_x, inner_y))
                
                inner_color_idx = (color_idx + 1) % len(colors)
                draw.polygon(inner_points, fill=colors[inner_color_idx])
            
            frames.append(img)
        
        metadata = {
            'type': 'morphing_shapes',
            'num_vertices': num_vertices,
            'morph_speed': morph_speed,
            'base_size': base_size
        }
        
        return frames, metadata


def sequence_collate_fn(batch):
    """Custom collate function for sequence dataset with metadata"""
    frames_batch = []
    metadata_batch = []
    
    for frames, metadata in batch:
        frames_batch.append(frames)
        metadata_batch.append(metadata)
    
    # Stack frames tensors
    frames_tensor = torch.stack(frames_batch, dim=0)
    
    return frames_tensor, metadata_batch


def create_sequence_dataloader(
    batch_size: int = 4,
    sequence_length: int = 32,
    num_sequences: int = 1000,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 2,
    physics_enabled: bool = True
):
    """Create DataLoader for sequence generation training"""
    
    dataset = LongSequenceDataset(
        sequence_length=sequence_length,
        num_sequences=num_sequences,
        image_size=image_size,
        physics_enabled=physics_enabled
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=sequence_collate_fn
    )
    
    return dataloader


def test_sequence_dataset():
    """Test the sequence dataset"""
    print("[TEST] Testing LongSequenceDataset")
    
    dataset = LongSequenceDataset(
        sequence_length=16,
        num_sequences=5,
        image_size=(128, 128)
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Test single sequence
    frames, metadata = dataset[0]
    print(f"   Sequence shape: {frames.shape}")
    print(f"   Sequence type: {metadata['scene_type']}")
    print(f"   Metadata: {list(metadata.keys())}")
    
    # Test dataloader
    dataloader = create_sequence_dataloader(
        batch_size=2,
        sequence_length=8,
        num_sequences=10,
        image_size=(64, 64),
        num_workers=0  # No multiprocessing for test
    )
    
    print(f"   Dataloader batches: {len(dataloader)}")
    
    for batch_frames, batch_metadata in dataloader:
        print(f"   Batch shape: {batch_frames.shape}")
        print(f"   Batch types: {[m['scene_type'] for m in batch_metadata]}")
        break
    
    print("[SUCCESS] Sequence dataset test completed!")


if __name__ == "__main__":
    test_sequence_dataset()