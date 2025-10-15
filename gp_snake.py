import random
import copy
from typing import List
import numpy as np
from snake_environment import SnakeGame, Direction
import time
import pickle
import json
import logging

# Configure logging to write to a file
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Node base class for GP tree
class Node:
    def evaluate(self, state: np.ndarray):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def get_subnodes(self) -> List['Node']:
        return []
    
    def to_dict(self):
        """Convert node to dictionary for serialization"""
        raise NotImplementedError
    
    @staticmethod
    def from_dict(data):
        """Create node from dictionary"""
        if data['type'] == 'ActionNode':
            return ActionNode.from_dict(data)
        elif data['type'] == 'ConditionNode':
            return ConditionNode.from_dict(data)
        else:
            raise ValueError(f"Unknown node type: {data['type']}")

# Leaf node that returns an action
class ActionNode(Node):
    def __init__(self, action: Direction):
        self.action = action

    def evaluate(self, state: np.ndarray):
        return self.action

    def copy(self):
        return ActionNode(self.action)

    def __str__(self):
        return f"Action({self.action.name})"
    
    def to_dict(self):
        return {
            'type': 'ActionNode',
            'action': self.action.value
        }
    
    @staticmethod
    def from_dict(data):
        return ActionNode(Direction(data['action']))

# Conditional node that splits on a state feature
class ConditionNode(Node):
    def __init__(self, feature_idx: int, threshold: float, left: Node, right: Node):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

    def evaluate(self, state: np.ndarray):
        if state[self.feature_idx] > self.threshold:
            return self.left.evaluate(state)
        else:
            return self.right.evaluate(state)

    def copy(self):
        return ConditionNode(
            self.feature_idx,
            self.threshold,
            self.left.copy(),
            self.right.copy()
        )

    def get_subnodes(self) -> List[Node]:
        nodes = [self]
        nodes += self.left.get_subnodes()
        nodes += self.right.get_subnodes()
        return nodes

    def __str__(self):
        return f"If state[{self.feature_idx}] > {self.threshold:.2f} then ({self.left}) else ({self.right})"
    
    def to_dict(self):
        return {
            'type': 'ConditionNode',
            'feature_idx': self.feature_idx,
            'threshold': self.threshold,
            'left': self.left.to_dict(),
            'right': self.right.to_dict()
        }
    
    @staticmethod
    def from_dict(data):
        left = Node.from_dict(data['left'])
        right = Node.from_dict(data['right'])
        return ConditionNode(data['feature_idx'], data['threshold'], left, right)

# Generate a random tree with better balance for initial population
def generate_random_tree(max_depth: int, action_choices: List[Direction]) -> Node:
    # If at or below minimum depth, create a leaf action node
    if max_depth <= 1:
        return ActionNode(random.choice(action_choices))
    # Increased early stopping probability for smaller trees
    if random.random() < 0.4:  # Increased from 0.2
        return ActionNode(random.choice(action_choices))
    
    # Focus on more meaningful features
    # Features 0-3: danger detection, 4-7: direction, 8-9: food direction, 10: snake length
    important_features = [0, 1, 2, 3, 8, 9]  # Danger and food direction are most important
    feature_idx = random.choice(important_features) if random.random() < 0.7 else random.randint(0, 10)
    
    # Better threshold selection based on feature type
    if feature_idx in [0, 1, 2, 3]:  # Danger features (binary)
        threshold = 0.5
    elif feature_idx in [4, 5, 6, 7]:  # Direction features (binary)
        threshold = 0.5
    elif feature_idx in [8, 9]:  # Food direction (-1 to 1)
        threshold = random.uniform(-0.5, 0.5)
    else:  # Snake length (0 to 1)
        threshold = random.uniform(0.1, 0.9)
    
    left = generate_random_tree(max_depth - 1, action_choices)
    right = generate_random_tree(max_depth - 1, action_choices)
    return ConditionNode(feature_idx, threshold, left, right)

# Swap subtrees between two trees
def crossover(tree1: Node, tree2: Node) -> Node:
    t1 = tree1.copy()
    t2 = tree2.copy()
    # collect interior nodes for possible subtree swap
    subs1 = [n for n in t1.get_subnodes() if isinstance(n, ConditionNode)]
    subs2 = [n for n in t2.get_subnodes() if isinstance(n, ConditionNode)]
    if not subs1 or not subs2:
        return t1
    n1 = random.choice(subs1)
    n2 = random.choice(subs2)
    # swap branches
    n1.feature_idx, n1.threshold, n1.left, n1.right, n2.feature_idx, n2.threshold, n2.left, n2.right = (
        n2.feature_idx, n2.threshold, n2.left, n2.right,
        n1.feature_idx, n1.threshold, n1.left, n1.right
    )
    return t1

# Mutate nodes randomly
def mutate(tree: Node, mutation_rate: float, max_depth: int, action_choices: List[Direction]) -> Node:
    def _mutate(node: Node, depth: int) -> Node:
        if random.random() < mutation_rate:
            return generate_random_tree(max_depth - depth, action_choices)
        if isinstance(node, ConditionNode):
            node.left = _mutate(node.left, depth + 1)
            node.right = _mutate(node.right, depth + 1)
        return node
    return _mutate(tree.copy(), 0)

# Genetic Program class to evolve trees
class GeneticProgram:
    def __init__(   
        self,
        pop_size: int = 100,
        generations: int = 100,  # Reduced from 2000
        max_depth: int = 4,     # Reduced from 7
        mutation_rate: float = 0.4,   # Increased from 0.2
        crossover_rate: float = 0.6,  # Reduced from 0.8
        elite_size: int = 15,     # New: elitism
        diversity_threshold: int = 5  # New: diversity enforcement
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.diversity_threshold = diversity_threshold
        self.action_choices = list(Direction)
        self.population: List[Node] = [
            generate_random_tree(self.max_depth, self.action_choices)
            for _ in range(self.pop_size)
        ]
        self.stagnation_counter = 0
        self.last_best_fitness = -float('inf')

    # Fitness: combine score and survival steps with more nuanced evaluation
    def evaluate(self, tree: Node, episodes: int = 5) -> float:
        total_fitness = 0.0
        for _ in range(episodes):
            game = SnakeGame(width=20, height=20, render=False)  # Smaller board for faster learning
            state = game.get_state_vector()
            steps_survived = 0
            food_collected = 0
            
            while not game.game_over:
                action = tree.evaluate(state)
                old_distance = game.get_manhattan_distance_to_food()
                _, reward, done, info = game.step(action)
                new_distance = game.get_manhattan_distance_to_food()
                
                steps_survived += 1
                if game.score > food_collected:
                    food_collected = game.score
                
                state = game.get_state_vector()
            
            # Multi-objective fitness: prioritize food collection, then survival
            episode_fitness = (
                food_collected * 2000 +      # Heavy weight on food
                steps_survived * 1 +         # Survival bonus
                (100 if food_collected > 0 else 0)  # Bonus for any food
            )
            total_fitness += episode_fitness
            
        return total_fitness / episodes

    # Tournament selection with diversity pressure
    def select(self, fitnesses: List[float]) -> List[Node]:
        selected = []
        for _ in range(self.pop_size):
            # Tournament selection with larger tournament size
            tournament_size = min(5, self.pop_size)
            tournament_indices = random.sample(range(self.pop_size), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            selected.append(self.population[winner_idx])
        return selected

    def inject_diversity(self):
        """Replace worst individuals with random ones to maintain diversity"""
        # Sort population by fitness
        fitnesses = [self.evaluate(t, episodes=1) for t in self.population]
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        
        # Replace bottom 20% with new random individuals
        num_to_replace = max(1, self.pop_size // 5)
        for i in range(num_to_replace):
            idx = sorted_indices[i]
            self.population[idx] = generate_random_tree(self.max_depth, self.action_choices)

    # Main evolution loop with adaptive parameters and early stopping
    def run(self) -> Node:
        # Initialize best_tree to first individual to satisfy return type
        best_tree = self.population[0].copy()
        best_fitness = -float('inf')
        
        for gen in range(self.generations):
            fitnesses = [self.evaluate(t) for t in self.population]
            
            # Track best individual
            current_best_fitness = max(fitnesses)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_idx = fitnesses.index(current_best_fitness)
                best_tree = self.population[best_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            print(f"Generation {gen+1}/{self.generations} - Best fitness: {best_fitness:.2f} - Avg: {np.mean(fitnesses):.2f}")

            # Inject diversity if stagnating
            if self.stagnation_counter > 0 and self.stagnation_counter % self.diversity_threshold == 0:
                print(f"Injecting diversity at generation {gen+1}")
                self.inject_diversity()
            
            
            # Elitism: keep best individuals
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:self.elite_size]
            elites = [self.population[i].copy() for i in elite_indices]
            
            # Selection
            parents = self.select(fitnesses)
            
            # Crossover and Mutation
            children: List[Node] = []
            for i in range(0, self.pop_size - self.elite_size, 2):
                p1 = parents[i % len(parents)]
                p2 = parents[(i+1) % len(parents)]
                
                if random.random() < self.crossover_rate:
                    c1 = crossover(p1, p2)
                    c2 = crossover(p2, p1)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Apply adaptive mutation
                c1 = mutate(c1, self.mutation_rate, self.max_depth, self.action_choices)
                c2 = mutate(c2, self.mutation_rate, self.max_depth, self.action_choices)
                
                children.extend([c1, c2])
            
            # Combine elites and children for next generation
            self.population = elites + children[:self.pop_size - self.elite_size]
            
        return best_tree

# Utility functions to train and test

def save_gp_agent(tree: Node, filename: str):
    """Save GP agent to file"""
    data = tree.to_dict()
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"GP agent saved to {filename}")

def load_gp_agent(filename: str) -> Node:
    """Load GP agent from file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    tree = Node.from_dict(data)
    print(f"GP agent loaded from {filename}")
    return tree

def train_gp_agent() -> Node:
    gp = GeneticProgram()
    best = gp.run()
    print(f"Trained GP agent tree:\n{best}")

    # Save the best agent
    save_gp_agent(best, "best_gp_agent.json")

    return best


def test_gp_agent(tree: Node, episodes: int = 5):
    for ep in range(episodes):
        # Run headless for automated testing
        game = SnakeGame(render=False)
        state = game.get_state_vector()
        while not game.game_over:
            action = tree.evaluate(state)
            # perform step and update state vector
            _, reward, done, info = game.step(action)
            state = game.get_state_vector()
            # no rendering for headless test
        print(f"Episode {ep+1}: Score={game.score}, Steps={game.steps}")
        game.close()

def visualize_gp_agent(tree: Node):
    """
    Run a single visualization of the GP agent playing with rendering enabled.
    """
    game = SnakeGame(render=True)
    state = game.get_state_vector()
    while not game.game_over:
        action = tree.evaluate(state)
        # step and update state
        _, _, _, _ = game.step(action)
        state = game.get_state_vector()
        game.render()
        time.sleep(0.1)
    print(f"Visual playback: Score={game.score}, Steps={game.steps}")
    game.close()

# Modify the main block to skip visualization
if __name__ == "__main__":
    best_tree = train_gp_agent()
    test_gp_agent(best_tree)
    visualize_gp_agent(best_tree)
