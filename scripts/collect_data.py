#!/usr/bin/env python3
"""
Automated Data Collection Pipeline for World Model Training
Collects gameplay data from multiple sources:
1. OpenGameArt assets
2. itch.io open-source games  
3. GitHub game repositories
4. Procedural generation
"""

import os
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import cv2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import torch
from stable_baselines3 import PPO, A2C
import gymnasium as gym
from tqdm import tqdm
import hashlib
import zipfile
import git


class DataCollector:
    def __init__(self, output_dir: str = "datasets/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_frames': 0,
            'total_sequences': 0,
            'total_games': 0,
            'sources': {}
        }
        
    def collect_all(self, target_hours: int = 1000):
        """Collect data from all sources until target hours reached"""
        target_frames = target_hours * 3600 * 30  # 30 FPS
        
        collectors = [
            self.collect_opengameart,
            self.collect_itch_games,
            self.collect_github_games,
            self.generate_procedural
        ]
        
        for collector in collectors:
            if self.stats['total_frames'] >= target_frames:
                break
            collector(target_frames - self.stats['total_frames'])
        
        self.save_stats()
        print(f"Collection complete: {self.stats['total_frames']} frames collected")


class OpenGameArtCollector(DataCollector):
    """Collect sprite sheets and assets from OpenGameArt"""
    
    def collect_opengameart(self, target_frames: int):
        print("Collecting from OpenGameArt...")
        
        # Categories to download
        categories = [
            'sprites', 'backgrounds', 'tilesets', 
            'characters', 'enemies', 'items'
        ]
        
        base_url = "https://opengameart.org/api"
        collected = 0
        
        for category in categories:
            page = 1
            while collected < target_frames // len(categories):
                # API call to get assets
                response = requests.get(
                    f"{base_url}/content",
                    params={'type': category, 'page': page}
                )
                
                if response.status_code != 200:
                    break
                
                assets = response.json()
                
                for asset in assets:
                    # Download asset files
                    asset_dir = self.output_dir / 'opengameart' / asset['id']
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    
                    for file_url in asset.get('files', []):
                        self._download_file(file_url, asset_dir)
                    
                    # Generate synthetic sequences from sprites
                    sequences = self._generate_sprite_sequences(asset_dir)
                    collected += len(sequences) * 32  # 32 frames per sequence
                
                page += 1
        
        self.stats['sources']['opengameart'] = collected
        return collected
    
    def _generate_sprite_sequences(self, sprite_dir: Path) -> List[np.ndarray]:
        """Create animation sequences from sprite sheets"""
        sequences = []
        
        for sprite_file in sprite_dir.glob("*.png"):
            try:
                sprite_sheet = Image.open(sprite_file)
                
                # Detect sprite grid
                sprites = self._extract_sprites(sprite_sheet)
                
                if len(sprites) > 4:
                    # Create walking sequence
                    sequence = self._animate_sprites(sprites, 'walk')
                    sequences.append(sequence)
                    
                    # Create jump sequence
                    sequence = self._animate_sprites(sprites, 'jump')
                    sequences.append(sequence)
                    
            except Exception as e:
                print(f"Error processing {sprite_file}: {e}")
        
        return sequences
    
    def _extract_sprites(self, sheet: Image) -> List[np.ndarray]:
        """Extract individual sprites from sheet"""
        # Simple grid detection (can be improved)
        sheet_array = np.array(sheet)
        h, w = sheet_array.shape[:2]
        
        # Assume 32x32 or 64x64 sprites
        sprite_size = 32 if w % 32 == 0 else 64
        
        sprites = []
        for y in range(0, h, sprite_size):
            for x in range(0, w, sprite_size):
                sprite = sheet_array[y:y+sprite_size, x:x+sprite_size]
                if sprite.mean() > 0:  # Non-empty sprite
                    sprites.append(sprite)
        
        return sprites


class ItchGameCollector(DataCollector):
    """Collect gameplay from itch.io HTML5 games"""
    
    def collect_itch_games(self, target_frames: int):
        print("Collecting from itch.io games...")
        
        # Setup headless browser
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        
        # Find playable web games
        games = self._find_web_games()
        collected = 0
        
        for game_url in games:
            if collected >= target_frames:
                break
                
            try:
                # Load game
                driver.get(game_url)
                time.sleep(5)  # Let game load
                
                # Setup AI agent to play
                agent = self._setup_game_agent(driver)
                
                # Record gameplay
                frames = self._record_gameplay(driver, agent, duration=60)
                
                # Save sequence
                sequence_path = self.output_dir / 'itch' / f"{hashlib.md5(game_url.encode()).hexdigest()}.npz"
                np.savez_compressed(sequence_path, frames=frames['frames'], actions=frames['actions'])
                
                collected += len(frames['frames'])
                self.stats['total_games'] += 1
                
            except Exception as e:
                print(f"Error with game {game_url}: {e}")
        
        driver.quit()
        self.stats['sources']['itch'] = collected
        return collected
    
    def _find_web_games(self) -> List[str]:
        """Scrape itch.io for playable HTML5 games"""
        games = []
        
        # Categories likely to have 2D games
        categories = ['platformer', 'puzzle', 'arcade', 'action']
        
        for category in categories:
            url = f"https://itch.io/games/html5/tag-{category}"
            response = requests.get(url)
            
            # Parse HTML to find game links (simplified)
            # In production, use BeautifulSoup
            import re
            pattern = r'href="(https://[\w\-]+\.itch\.io/[\w\-]+)"'
            matches = re.findall(pattern, response.text)
            games.extend(matches[:10])  # Top 10 per category
        
        return games
    
    def _setup_game_agent(self, driver):
        """Create RL agent to play the game"""
        # Simplified - in reality would need game-specific setup
        class BrowserGameEnv(gym.Env):
            def __init__(self, driver):
                self.driver = driver
                self.action_space = gym.spaces.Discrete(5)  # up,down,left,right,action
                self.observation_space = gym.spaces.Box(0, 255, (256, 256, 3))
                
            def step(self, action):
                # Send action to browser
                key_map = {
                    0: Keys.UP,
                    1: Keys.DOWN,
                    2: Keys.LEFT,
                    3: Keys.RIGHT,
                    4: Keys.SPACE
                }
                
                body = self.driver.find_element(By.TAG_NAME, 'body')
                body.send_keys(key_map[action])
                
                # Get screenshot
                screenshot = self.driver.get_screenshot_as_png()
                obs = np.array(Image.open(io.BytesIO(screenshot)))
                obs = cv2.resize(obs, (256, 256))
                
                # Simple reward (game-specific in reality)
                reward = np.random.random()
                done = False
                
                return obs, reward, done, {}
                
            def reset(self):
                self.driver.refresh()
                time.sleep(3)
                return self.step(0)[0]
        
        env = BrowserGameEnv(driver)
        
        # Use pre-trained agent or random actions
        try:
            agent = PPO.load("models/game_player_ppo")
        except:
            # Random agent as fallback
            class RandomAgent:
                def predict(self, obs, deterministic=False):
                    return np.random.randint(0, 5), None
            agent = RandomAgent()
        
        return agent
    
    def _record_gameplay(self, driver, agent, duration: int = 60) -> Dict:
        """Record agent playing the game"""
        frames = []
        actions = []
        
        fps = 30
        total_frames = duration * fps
        
        for _ in tqdm(range(total_frames), desc="Recording gameplay"):
            # Get current frame
            screenshot = driver.get_screenshot_as_png()
            frame = np.array(Image.open(io.BytesIO(screenshot)))
            frame = cv2.resize(frame, (256, 256))
            frames.append(frame)
            
            # Get action from agent
            action, _ = agent.predict(frame)
            actions.append(action)
            
            # Execute action
            key_map = {0: Keys.UP, 1: Keys.DOWN, 2: Keys.LEFT, 3: Keys.RIGHT, 4: Keys.SPACE}
            body = driver.find_element(By.TAG_NAME, 'body')
            body.send_keys(key_map[action])
            
            time.sleep(1/fps)
        
        return {'frames': np.array(frames), 'actions': np.array(actions)}


class GitHubGameCollector(DataCollector):
    """Collect games from GitHub repositories"""
    
    def collect_github_games(self, target_frames: int):
        print("Collecting from GitHub games...")
        
        # Search for game repositories
        repos = self._search_game_repos()
        collected = 0
        
        for repo_url in repos:
            if collected >= target_frames:
                break
                
            try:
                # Clone repository
                repo_name = repo_url.split('/')[-1]
                repo_path = self.output_dir / 'github' / repo_name
                
                if not repo_path.exists():
                    git.Repo.clone_from(repo_url, repo_path)
                
                # Detect game engine
                engine = self._detect_engine(repo_path)
                
                # Run game and record
                if engine:
                    frames = self._run_and_record(repo_path, engine)
                    collected += len(frames)
                    
            except Exception as e:
                print(f"Error with repo {repo_url}: {e}")
        
        self.stats['sources']['github'] = collected
        return collected
    
    def _search_game_repos(self) -> List[str]:
        """Find game repositories on GitHub"""
        repos = []
        
        # GitHub API search
        headers = {'Accept': 'application/vnd.github.v3+json'}
        
        # Search queries for 2D games
        queries = [
            'language:javascript game 2d',
            'language:python pygame',
            'godot 2d platformer',
            'love2d game'
        ]
        
        for query in queries:
            response = requests.get(
                'https://api.github.com/search/repositories',
                params={'q': query, 'sort': 'stars', 'per_page': 10},
                headers=headers
            )
            
            if response.status_code == 200:
                for repo in response.json()['items']:
                    repos.append(repo['clone_url'])
        
        return repos
    
    def _detect_engine(self, repo_path: Path) -> str:
        """Detect game engine from repository files"""
        if (repo_path / 'package.json').exists():
            with open(repo_path / 'package.json') as f:
                data = json.load(f)
                if 'phaser' in str(data).lower():
                    return 'phaser'
        
        if (repo_path / 'project.godot').exists():
            return 'godot'
        
        if any(repo_path.glob('*.py')):
            for py_file in repo_path.glob('*.py'):
                with open(py_file) as f:
                    if 'pygame' in f.read():
                        return 'pygame'
        
        return None
    
    def _run_and_record(self, repo_path: Path, engine: str) -> np.ndarray:
        """Run game and record frames"""
        frames = []
        
        if engine == 'pygame':
            # Run pygame game with recording wrapper
            subprocess.run([
                'python', 'scripts/pygame_recorder.py',
                '--game', str(repo_path),
                '--output', str(self.output_dir / 'recordings')
            ])
            
        elif engine == 'godot':
            # Run Godot in headless mode with recording
            subprocess.run([
                'godot', '--headless', '--record',
                str(repo_path / 'project.godot')
            ])
            
        # Load recorded frames
        recording_file = self.output_dir / 'recordings' / 'latest.npz'
        if recording_file.exists():
            data = np.load(recording_file)
            frames = data['frames']
        
        return frames


class ProceduralGenerator(DataCollector):
    """Generate synthetic game data procedurally"""
    
    def generate_procedural(self, target_frames: int):
        print("Generating procedural data...")
        
        generators = [
            self._generate_platformer,
            self._generate_puzzle,
            self._generate_topdown
        ]
        
        collected = 0
        frames_per_type = target_frames // len(generators)
        
        for generator in generators:
            frames = generator(frames_per_type)
            collected += len(frames)
        
        self.stats['sources']['procedural'] = collected
        return collected
    
    def _generate_platformer(self, num_frames: int) -> np.ndarray:
        """Generate platformer game sequences"""
        sequences = []
        frames_per_sequence = 32
        num_sequences = num_frames // frames_per_sequence
        
        for _ in tqdm(range(num_sequences), desc="Generating platformers"):
            # Create level
            level = self._create_platformer_level()
            
            # Simulate physics
            sequence = self._simulate_platformer(level, frames_per_sequence)
            sequences.extend(sequence)
        
        return np.array(sequences)
    
    def _create_platformer_level(self) -> Dict:
        """Create a simple platformer level"""
        width, height = 256, 256
        
        level = {
            'platforms': [],
            'player': {'x': 32, 'y': 200, 'vx': 0, 'vy': 0},
            'enemies': [],
            'collectibles': []
        }
        
        # Add platforms
        for i in range(5):
            platform = {
                'x': np.random.randint(0, width-64),
                'y': np.random.randint(100, height-20),
                'width': np.random.randint(32, 96),
                'height': 16
            }
            level['platforms'].append(platform)
        
        # Add enemies
        for i in range(3):
            enemy = {
                'x': np.random.randint(0, width),
                'y': np.random.randint(0, height-32),
                'type': np.random.choice(['walker', 'jumper'])
            }
            level['enemies'].append(enemy)
        
        return level
    
    def _simulate_platformer(self, level: Dict, num_frames: int) -> List[np.ndarray]:
        """Simulate platformer physics and render frames"""
        frames = []
        
        for frame_idx in range(num_frames):
            # Update physics
            level['player']['vy'] += 0.5  # Gravity
            level['player']['y'] += level['player']['vy']
            level['player']['x'] += level['player']['vx']
            
            # Check collisions
            for platform in level['platforms']:
                if self._check_collision(level['player'], platform):
                    level['player']['y'] = platform['y'] - 16
                    level['player']['vy'] = 0
            
            # Render frame
            frame = self._render_level(level)
            frames.append(frame)
            
            # Simulate player actions
            if frame_idx % 10 == 0:
                action = np.random.choice(['left', 'right', 'jump'])
                if action == 'left':
                    level['player']['vx'] = -2
                elif action == 'right':
                    level['player']['vx'] = 2
                elif action == 'jump' and level['player']['vy'] == 0:
                    level['player']['vy'] = -10
        
        return frames
    
    def _render_level(self, level: Dict) -> np.ndarray:
        """Render level to image"""
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Draw platforms
        for platform in level['platforms']:
            x, y = int(platform['x']), int(platform['y'])
            w, h = platform['width'], platform['height']
            frame[y:y+h, x:x+w] = [139, 69, 19]  # Brown
        
        # Draw player
        px, py = int(level['player']['x']), int(level['player']['y'])
        frame[py:py+16, px:px+16] = [0, 255, 0]  # Green
        
        # Draw enemies
        for enemy in level['enemies']:
            ex, ey = int(enemy['x']), int(enemy['y'])
            frame[ey:ey+16, ex:ex+16] = [255, 0, 0]  # Red
        
        return frame
    
    def _check_collision(self, obj1: Dict, obj2: Dict) -> bool:
        """Simple AABB collision detection"""
        return (obj1['x'] < obj2['x'] + obj2.get('width', 16) and
                obj1['x'] + 16 > obj2['x'] and
                obj1['y'] < obj2['y'] + obj2.get('height', 16) and
                obj1['y'] + 16 > obj2['y'])


class DataProcessor:
    """Process and clean collected data"""
    
    def __init__(self, raw_dir: str = "datasets/raw", processed_dir: str = "datasets/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_all(self):
        """Process all raw data"""
        print("Processing collected data...")
        
        # Process each source
        for source_dir in self.raw_dir.iterdir():
            if source_dir.is_dir():
                self._process_source(source_dir)
        
        # Create train/val/test splits
        self._create_splits()
        
        # Generate metadata
        self._generate_metadata()
        
    def _process_source(self, source_dir: Path):
        """Process data from a specific source"""
        processed_sequences = []
        
        for data_file in source_dir.glob("**/*.npz"):
            try:
                data = np.load(data_file)
                
                # Clean and normalize
                frames = data['frames']
                frames = self._normalize_frames(frames)
                
                # Remove corrupted sequences
                if self._validate_sequence(frames):
                    processed_sequences.append({
                        'frames': frames,
                        'actions': data.get('actions', None),
                        'source': source_dir.name
                    })
                    
            except Exception as e:
                print(f"Error processing {data_file}: {e}")
        
        # Save processed data
        output_file = self.processed_dir / f"{source_dir.name}_processed.npz"
        np.savez_compressed(output_file, sequences=processed_sequences)
        
    def _normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Normalize frames to consistent format"""
        normalized = []
        
        for frame in frames:
            # Resize to 256x256 if needed
            if frame.shape[:2] != (256, 256):
                frame = cv2.resize(frame, (256, 256))
            
            # Convert to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            normalized.append(frame)
        
        return np.array(normalized)
    
    def _validate_sequence(self, frames: np.ndarray) -> bool:
        """Check if sequence is valid"""
        if len(frames) < 16:
            return False
        
        # Check for stuck/static sequences
        frame_diff = np.mean(np.abs(frames[1:] - frames[:-1]))
        if frame_diff < 0.01:  # Too static
            return False
        
        # Check for corrupted frames
        if np.any(np.isnan(frames)) or np.any(np.isinf(frames)):
            return False
        
        return True


def main():
    """Main data collection pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect data for world model training')
    parser.add_argument('--hours', type=int, default=100, help='Hours of gameplay to collect')
    parser.add_argument('--output', type=str, default='datasets', help='Output directory')
    parser.add_argument('--process', action='store_true', help='Process after collection')
    
    args = parser.parse_args()
    
    # Collect data
    collector = DataCollector(output_dir=f"{args.output}/raw")
    collector.collect_all(target_hours=args.hours)
    
    # Process if requested
    if args.process:
        processor = DataProcessor(
            raw_dir=f"{args.output}/raw",
            processed_dir=f"{args.output}/processed"
        )
        processor.process_all()
    
    print("Data collection complete!")


if __name__ == "__main__":
    main()