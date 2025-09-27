"""Deterministic grid-based toy world used when no trained model is available.

The goal is to provide a lightweight, fully deterministic environment so the
interactive API has meaningful behaviour even without learned weights.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

Tile = int
Position = Tuple[int, int]


@dataclass
class ToyWorldState:
    grid: np.ndarray  # (grid_size, grid_size) of tile ids
    player: Position
    goal: Position
    tick: int
    seed: int
    prompt: Optional[str]


class ToyWorldSimulator:
    """Generates and updates a tiny tile-based world.

    A session is represented by a 16x16 grid rendered to 256x256 RGB frames.
    Actions move a single avatar while avoiding obstacles.
    """

    def __init__(self, grid_size: int = 16, tile_size: int = 16) -> None:
        self.grid_size = grid_size
        self.tile_size = tile_size
        self._colors = self._build_color_map()
        self._action_map: Dict[int, Position] = {
            0: (0, 0),  # stay
            1: (-1, 0),  # up
            2: (1, 0),   # down
            3: (0, -1),  # left
            4: (0, 1),   # right
        }

    def create_state(self, prompt: Optional[str], seed: Optional[int]) -> ToyWorldState:
        rng_seed = self._derive_seed(prompt, seed)
        rng = np.random.default_rng(rng_seed)

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        # Place scattered obstacles/decoration away from centre
        obstacle_count = self.grid_size  # one per row on average
        for _ in range(obstacle_count):
            r, c = rng.integers(0, self.grid_size, size=2)
            if self._is_safe_zone(r, c):
                continue
            grid[r, c] = 1  # obstacle

        decoration_count = self.grid_size // 2
        for _ in range(decoration_count):
            r, c = rng.integers(0, self.grid_size, size=2)
            if grid[r, c] != 0:
                continue
            grid[r, c] = 3  # decoration

        player = (self.grid_size // 2, self.grid_size // 2)
        grid[player] = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = player[0] + dr, player[1] + dc
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    if grid[rr, cc] == 1:
                        grid[rr, cc] = 0

        goal = self._place_goal(rng, grid, player)

        return ToyWorldState(
            grid=grid,
            player=player,
            goal=goal,
            tick=0,
            seed=rng_seed,
            prompt=prompt,
        )

    def render(self, state: ToyWorldState) -> np.ndarray:
        tile_img = self._colors[state.grid]
        tile_img[state.goal] = np.array([243, 162, 20], dtype=np.uint8)
        tile_img[state.player] = np.array([214, 48, 49], dtype=np.uint8)

        img = np.repeat(tile_img, self.tile_size, axis=0)
        img = np.repeat(img, self.tile_size, axis=1)

        # Add subtle scanline effect that depends on tick to show time progression
        if state.tick % 2 == 1:
            img[::2] = (img[::2] * 0.95).astype(np.uint8)

        return img

    def step(self, state: ToyWorldState, action: int) -> Tuple[ToyWorldState, np.ndarray, Dict[str, object]]:
        move = self._action_map.get(action, (0, 0))
        target = (state.player[0] + move[0], state.player[1] + move[1])

        moved = False
        if self._is_walkable(state, target):
            state.player = target
            moved = True

        reached_goal = state.player == state.goal
        if reached_goal:
            rng = np.random.default_rng(state.seed + state.tick + 1)
            state.goal = self._place_goal(rng, state.grid, state.player)

        state.tick += 1
        frame = self.render(state)

        metrics = {
            "player_row": state.player[0],
            "player_col": state.player[1],
            "tick": state.tick,
            "moved": moved,
            "reached_goal": reached_goal,
        }
        return state, frame, metrics

    def _derive_seed(self, prompt: Optional[str], seed: Optional[int]) -> int:
        if seed is not None:
            return int(seed & 0xFFFFFFFF)
        if prompt:
            digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            return int(digest[:8], 16)
        return 0

    def _place_goal(self, rng: np.random.Generator, grid: np.ndarray, player: Position) -> Position:
        attempts = 0
        while attempts < 128:
            r, c = rng.integers(0, self.grid_size, size=2)
            if (r, c) == player:
                attempts += 1
                continue
            if grid[r, c] == 0:
                return (int(r), int(c))
            attempts += 1
        # Fallback: place goal opposite the player
        return (max(0, self.grid_size - player[0] - 1), max(0, self.grid_size - player[1] - 1))

    def _is_safe_zone(self, row: int, col: int) -> bool:
        centre = self.grid_size // 2
        return abs(row - centre) <= 1 and abs(col - centre) <= 1

    def _is_walkable(self, state: ToyWorldState, pos: Position) -> bool:
        r, c = pos
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        return state.grid[r, c] != 1

    def _build_color_map(self) -> np.ndarray:
        return np.array(
            [
                [33, 47, 60],     # empty floor
                [70, 70, 90],     # obstacle
                [240, 240, 240],  # reserved (unused)
                [32, 147, 211],   # decoration/water
            ],
            dtype=np.uint8,
        )
