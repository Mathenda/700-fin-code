import numpy as np
import heapq
from typing import List, Tuple, Optional, Set
import time
from snake_environment import SnakeGame, Direction

class AStarSnakeAgent:
    """
    A* pathfinding agent for Snake game.
    Uses A* algorithm to find optimal path to food while avoiding collisions.
    """
    
    def __init__(self):
        self.name = "A* Agent"
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, pos: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []
        
        # Up, Right, Down, Left
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < width and 0 <= new_y < height:
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def is_safe_move(self, pos: Tuple[int, int], snake_pos: List[Tuple[int, int]], 
                     future_snake_length: int) -> bool:
        """
        Check if a position is safe to move to.
        Takes into account that the tail will move if no food is eaten.
        """
        # Don't move into current snake body (except tail if we're not growing)
        snake_body = snake_pos[:-1] if future_snake_length == len(snake_pos) else snake_pos
        return pos not in snake_body
    
    def a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int], 
                      snake_pos: List[Tuple[int, int]], width: int, height: int) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm to find path from start to goal.
        Returns path as list of positions, or None if no path exists.
        """
        if start == goal:
            return [start]
        
        # Priority queue: (f_cost, g_cost, position, path)
        open_set = [(0, 0, start, [start])]
        closed_set: Set[Tuple[int, int]] = set()
        
        # For path reconstruction
        g_costs = {start: 0}
        
        while open_set:
            f_cost, g_cost, current_pos, path = heapq.heappop(open_set)
            
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            
            if current_pos == goal:
                return path
            
            # Check neighbors
            for neighbor in self.get_neighbors(current_pos, width, height):
                if neighbor in closed_set:
                    continue
                
                # Check if this move is safe
                future_length = len(snake_pos) + (1 if goal in path else 0)
                if not self.is_safe_move(neighbor, snake_pos, future_length):
                    continue
                
                tentative_g = g_cost + 1
                
                if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g
                    h_cost = self.manhattan_distance(neighbor, goal)
                    f_cost_new = tentative_g + h_cost
                    
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (f_cost_new, tentative_g, neighbor, new_path))
        
        return None  # No path found
    
    def get_direction_from_positions(self, from_pos: Tuple[int, int], 
                                   to_pos: Tuple[int, int]) -> Direction:
        """Convert two positions to a direction"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx == 1:
            return Direction.RIGHT
        elif dx == -1:
            return Direction.LEFT
        elif dy == 1:
            return Direction.DOWN
        elif dy == -1:
            return Direction.UP
        else:
            return Direction.RIGHT  # Default fallback
    
    def find_safe_move(self, game: SnakeGame) -> Direction:
        """
        Find a safe move when no path to food exists.
        Try to move to the largest open area or follow tail.
        """
        head = game.snake_pos[0]
        possible_moves = []
        
        # Try each direction
        for direction in Direction:
            if not game.is_valid_action(direction):
                continue
            
            # Simulate the move
            game_copy = game.copy()
            _, reward, done, _ = game_copy.step(direction)
            
            if not done:
                # Calculate open space accessible from this position
                open_space = self.calculate_accessible_space(game_copy.snake_pos[0], game_copy)
                possible_moves.append((direction, open_space, reward))
        
        if not possible_moves:
            # No safe moves, return any valid direction
            for direction in Direction:
                if game.is_valid_action(direction):
                    return direction
            return Direction.RIGHT  # Ultimate fallback
        
        # Choose move with largest accessible space
        possible_moves.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return possible_moves[0][0]
    
    def calculate_accessible_space(self, start: Tuple[int, int], game: SnakeGame) -> int:
        """
        Calculate how many cells are accessible from start position using BFS.
        This helps avoid getting trapped.
        """
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            current = queue.pop(0)
            
            for neighbor in self.get_neighbors(current, game.width, game.height):
                if neighbor in visited or neighbor in game.snake_pos:
                    continue
                
                visited.add(neighbor)
                queue.append(neighbor)
        
        return len(visited)
    
    def get_action(self, game: SnakeGame) -> Direction:
        """
        Get the next action for the agent using A* pathfinding.
        """
        if game.game_over or not game.snake_pos or not game.food_pos:
            return Direction.RIGHT
        
        head = game.snake_pos[0]
        food = game.food_pos
        
        # Find path to food using A*
        path = self.a_star_search(head, food, game.snake_pos, game.width, game.height)
        
        if path and len(path) > 1:
            # Follow the path
            next_pos = path[1]
            return self.get_direction_from_positions(head, next_pos)
        else:
            # No direct path to food, find safe move
            return self.find_safe_move(game)

def run_astar_game(episodes: int = 1, render: bool = True, width: int = 12, height: int = 12) -> List[dict]:
    """
    Run the A* agent for specified number of episodes.
    Returns statistics for each episode.
    """
    agent = AStarSnakeAgent()
    results = []
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        game = SnakeGame(width, height, render=render)
        episode_start_time = time.time()
        
        # Episode statistics
        total_reward = 0
        steps = 0
        max_score = 0
        
        while not game.game_over:
            if render:
                # Handle pygame events
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()
                        return results
            
            # Get action from A* agent
            action = agent.get_action(game)
            
            # Execute action
            state, reward, done, info = game.step(action)
            
            total_reward += reward
            steps += 1
            max_score = max(max_score, game.score)
            
            if render:
                game.render()
                time.sleep(0.05)  # Slow down for visibility
            
            # Safety check
            if steps > game.max_steps:
                break
        
        episode_time = time.time() - episode_start_time
        
        # Record episode results
        episode_result = {
            'episode': episode + 1,
            'score': game.score,
            'steps': steps,
            'total_reward': total_reward,
            'time': episode_time,
            'reason': info.get('reason', 'unknown'),
            'efficiency': game.score / steps if steps > 0 else 0,
            'snake_length': len(game.snake_pos)
        }
        
        results.append(episode_result)
        
        print(f"Episode {episode + 1} Results:")
        print(f"  Score: {game.score}")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Time: {episode_time:.2f}s")
        print(f"  End Reason: {info.get('reason', 'unknown')}")
        print(f"  Efficiency: {episode_result['efficiency']:.4f}")
        
        game.close()
    
    return results

def analyze_astar_results(results: List[dict]):
    """Analyze and print statistics from A* results"""
    if not results:
        return
    
    print("\n" + "="*50)
    print("A* PATHFINDING AGENT ANALYSIS")
    print("="*50)
    
    scores = [r['score'] for r in results]
    steps = [r['steps'] for r in results]
    rewards = [r['total_reward'] for r in results]
    times = [r['time'] for r in results]
    efficiencies = [r['efficiency'] for r in results]
    
    print(f"Episodes: {len(results)}")
    print(f"\nScore Statistics:")
    print(f"  Average: {np.mean(scores):.2f}")
    print(f"  Max: {np.max(scores)}")
    print(f"  Min: {np.min(scores)}")
    print(f"  Std: {np.std(scores):.2f}")
    
    print(f"\nSteps Statistics:")
    print(f"  Average: {np.mean(steps):.2f}")
    print(f"  Max: {np.max(steps)}")
    print(f"  Min: {np.min(steps)}")
    
    print(f"\nReward Statistics:")
    print(f"  Average: {np.mean(rewards):.2f}")
    print(f"  Max: {np.max(rewards):.2f}")
    print(f"  Min: {np.min(rewards):.2f}")
    
    print(f"\nTime Statistics:")
    print(f"  Average: {np.mean(times):.2f}s")
    print(f"  Total: {np.sum(times):.2f}s")
    
    print(f"\nEfficiency (Score/Steps):")
    print(f"  Average: {np.mean(efficiencies):.4f}")
    print(f"  Max: {np.max(efficiencies):.4f}")
    
    # End reasons analysis
    reasons = {}
    for result in results:
        reason = result['reason']
        reasons[reason] = reasons.get(reason, 0) + 1
    
    print(f"\nGame End Reasons:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count} ({count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    # Test A* agent
    print("Testing A* Snake Agent...")
    
    # Run a few episodes with rendering
    results = run_astar_game(episodes=3, render=True, width=10, height=10)
    analyze_astar_results(results)
    
    print("\nA* Agent test completed!")