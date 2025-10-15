import pygame
import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Optional
import time
from collections import deque

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeGame:
    """
    Shared Snake Game Environment for both A* and DQN implementations.
    Ensures fair comparison by providing identical game mechanics and state representation.
    """
    
    def __init__(self, width: int = 20, height: int = 20, cell_size: int = 20, render: bool = True):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.render_enabled = render
        
        # Game state
        self.snake_pos = []
        self.food_pos = None
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.max_steps = width * height * 2000  # Prevent infinite games
        # Loop detection setup
        self.head_history = deque(maxlen=50)
        self.loop_counter = 0
        self.loop_limit = 20000  # steps before considering a loop

        # Pygame setup for rendering
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
            
            # Colors
            self.BLACK = (0, 0, 0)
            self.GREEN = (0, 255, 0)
            self.RED = (255, 0, 0)
            self.WHITE = (255, 255, 255)
            self.BLUE = (0, 0, 255)
        
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        # Initialize snake at center
        center_x, center_y = self.width // 2, self.height // 2
        self.snake_pos = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_over = False
        self.steps = 0
        
        # Place first food
        self._place_food()
        
        return self.get_state()
    
    def _place_food(self):
        """Place food at random location not occupied by snake"""
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake_pos:
                self.food_pos = (x, y)
                break
    
    def is_valid_action(self, action: Direction) -> bool:
        """Check if the action is valid (not opposite to current direction)"""
        if len(self.snake_pos) <= 1:
            return True

        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return action != opposite[self.direction]

    def step(self, action: Direction) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one game step with the given action.
        Returns: (next_state, reward, done, info)
        """
        if self.game_over:
            return self.get_state(), 0.0, True, {"reason": "game_over"}

            # Validate action: End the game if the action is invalid.
        if not self.is_valid_action(action):
            self.game_over = True
            reward = -100.0
            done = True
            info = {"reason": "invalid_action_collision"}
            return self.get_state(), reward, done, info

        self.steps += 1

        # Validate action
        if not self.is_valid_action(action):
            action = self.direction
        self.direction = action

        # Calculate new head position
        head_x, head_y = self.snake_pos[0]
        if action == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif action == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif action == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        else:
            new_head = (head_x + 1, head_y)

        # Loop detection
        if new_head in self.head_history:
            self.loop_counter += 1
        else:
            self.loop_counter = 0
        self.head_history.append(new_head)
        if self.loop_counter >= self.loop_limit:
            self.game_over = True
            return self.get_state(), -100.0, True, {"reason": "loop_detected"}

        # Initialize reward and status
        reward = 0.0
        done = False
        info = {}

        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            self.game_over = True
            reward = -100.0
            done = True
            info["reason"] = "wall_collision"
        # Self collision
        elif new_head in self.snake_pos:
            self.game_over = True
            reward = -100.0
            done = True
            info["reason"] = "self_collision"
        # Food eaten
        elif new_head == self.food_pos:
            self.snake_pos.insert(0, new_head)
            self.score += 1
            reward = 50.0

            if len(self.snake_pos) == self.width * self.height:
                self.game_over = True
                reward += 100.0  
                done = True
                info["reason"] = "win"
                return self.get_state(), reward, done, info

            self._place_food()
        else:
            self.snake_pos.insert(0, new_head)
            self.snake_pos.pop()
            reward = -0.1

        # Max steps reached
        if self.steps >= self.max_steps:
            self.game_over = True
            done = True
            reward -= 50.0
            info["reason"] = "max_steps"

        # Distance reward
        if not done and self.food_pos is not None:
            head_x, head_y = self.snake_pos[0]
            food_x, food_y = self.food_pos
            dist = abs(head_x - food_x) + abs(head_y - food_y)
            if dist > 0:
                reward += 1.0 / dist

        return self.get_state(), reward, done, info
    
    def get_state(self) -> np.ndarray:
        """
        Get current game state as numpy array.
        Returns a multi-channel representation:
        - Channel 0: Snake body (1 for snake, 0 for empty)
        - Channel 1: Snake head (1 for head, 0 for other)
        - Channel 2: Food position (1 for food, 0 for other)
        - Channel 3: Possible collision areas (1 for dangerous, 0 for safe)
        """
        state = np.zeros((4, self.height, self.width), dtype=np.float32)
        
        # Snake body
        for pos in self.snake_pos:
            if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
                state[0, pos[1], pos[0]] = 1.0
        
        # Snake head
        if self.snake_pos:
            head_x, head_y = self.snake_pos[0]
            if 0 <= head_x < self.width and 0 <= head_y < self.height:
                state[1, head_y, head_x] = 1.0
        
        # Food position
        if self.food_pos:
            food_x, food_y = self.food_pos
            if 0 <= food_x < self.width and 0 <= food_y < self.height:
                state[2, food_y, food_x] = 1.0
        
        # Danger areas (walls and snake body)
        for y in range(self.height):
            for x in range(self.width):
                # Wall danger or snake collision danger
                if (x, y) in self.snake_pos:
                    state[3, y, x] = 1.0
        
        return state
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get simplified state vector for algorithms that need 1D input.
        Returns 11 values:
        - 4 danger indicators (up, right, down, left)
        - 4 direction indicators for current direction
        - 2 food direction indicators (relative to head)
        - 1 snake length indicator (normalized)
        """
        if not self.snake_pos:
            return np.zeros(11)
        
        head_x, head_y = self.snake_pos[0]
        state_vector = np.zeros(11)
        
        # Danger in each direction (0-3)
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        for i, (dx, dy) in enumerate(directions):
            next_x, next_y = head_x + dx, head_y + dy
            # Check wall collision or self collision
            if (next_x < 0 or next_x >= self.width or 
                next_y < 0 or next_y >= self.height or
                (next_x, next_y) in self.snake_pos):
                state_vector[i] = 1.0
        
        # Current direction (4-7)
        state_vector[4 + self.direction.value] = 1.0
        
        # Food direction relative to head (8-9)
        if self.food_pos:
            food_x, food_y = self.food_pos
            if food_x > head_x:
                state_vector[8] = 1.0  # Food to the right
            elif food_x < head_x:
                state_vector[8] = -1.0  # Food to the left
            
            if food_y > head_y:
                state_vector[9] = 1.0  # Food below
            elif food_y < head_y:
                state_vector[9] = -1.0  # Food above
        
        # Snake length (normalized) (10)
        state_vector[10] = len(self.snake_pos) / (self.width * self.height)
        
        return state_vector
    
    def get_manhattan_distance_to_food(self) -> int:
        """Get Manhattan distance from snake head to food"""
        if not self.snake_pos or not self.food_pos:
            return 0
        
        head_x, head_y = self.snake_pos[0]
        food_x, food_y = self.food_pos
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def render(self):
        """Render the game using pygame"""
        if not self.render_enabled:
            return
        
        self.screen.fill(self.BLACK)
        
        # Draw snake
        for i, pos in enumerate(self.snake_pos):
            x, y = pos
            color = self.GREEN if i > 0 else self.BLUE  # Head is blue, body is green
            pygame.draw.rect(
                self.screen, 
                color,
                (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            )
            
            # Add border to snake segments
            pygame.draw.rect(
                self.screen,
                self.WHITE,
                (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                1
            )
        
        # Draw food
        if self.food_pos:
            x, y = self.food_pos
            pygame.draw.rect(
                self.screen,
                self.RED,
                (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            )
        
        # Display score and steps
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.WHITE)
        steps_text = font.render(f"Steps: {self.steps}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 50))
        
        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS for visibility
    
    def close(self):
        """Clean up pygame resources"""
        if self.render_enabled:
            pygame.quit()
    
    def copy(self):
        """Create a copy of the current game state for simulation purposes"""
        new_game = SnakeGame(self.width, self.height, self.cell_size, render=False)
        new_game.snake_pos = self.snake_pos.copy()
        new_game.food_pos = self.food_pos
        new_game.direction = self.direction
        new_game.score = self.score
        new_game.game_over = self.game_over
        new_game.steps = self.steps
        return new_game

def test_environment():
    """Test the snake environment"""
    print("Testing Snake Environment...")
    
    game = SnakeGame(10, 10, 30, render=True)
    
    # Test basic functionality
    print(f"Initial state shape: {game.get_state().shape}")
    print(f"Initial state vector shape: {game.get_state_vector().shape}")
    print(f"Initial score: {game.score}")
    print(f"Initial snake position: {game.snake_pos}")
    print(f"Initial food position: {game.food_pos}")
    
    # Test a few random moves
    actions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    
    for i in range(20):
        action = random.choice(actions)
        state, reward, done, info = game.step(action)
        
        print(f"Step {i+1}: Action={action.name}, Reward={reward:.2f}, Done={done}")
        if done:
            print(f"Game ended: {info}")
            break
        
        game.render()
        time.sleep(0.2)  # Slow down for visual inspection
    
    game.close()
    print("Environment test completed!")

if __name__ == "__main__":
    test_environment()