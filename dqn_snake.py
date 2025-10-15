import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from snake_environment import SnakeGame, Direction

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Snake game.
    Takes game state and outputs Q-values for each action.
    Optimized architecture for faster training.
    """
    
    def __init__(self, input_size: int = 11, hidden_sizes: List[int] = [128, 64], output_size: int = 4):
        super(DQNNetwork, self).__init__()
        
        # Create network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            # Reduced dropout for faster training and less regularization
            layers.append(nn.Dropout(0.05))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """
    Deep Q-Network Agent for Snake game with experience replay and target network.
    """
    
    def __init__(self, state_size: int = 11, action_size: int = 4, 
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 32, target_update_frequency: int = 100):
        
        self.name = "DQN Agent"
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Neural networks - optimized architecture
        self.network_arch = [128, 64]  # Smaller, more efficient architecture
        self.q_network = DQNNetwork(state_size, self.network_arch, action_size).to(device)
        self.target_network = DQNNetwork(state_size, self.network_arch, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.steps_done = 0
        self.update_count = 0
        self.losses = []
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def get_action(self, state, game: SnakeGame) -> Direction:
        """
        Choose action using epsilon-greedy policy.
        """
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        else:
            state_tensor = state
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action from valid actions
            valid_actions = []
            for action in Direction:
                if game.is_valid_action(action):
                    valid_actions.append(action)
            
            if valid_actions:
                return random.choice(valid_actions)
            else:
                return Direction.RIGHT  # Fallback
        else:
            # Greedy action selection
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
                # Mask invalid actions with very negative values
                masked_q_values = q_values.clone()
                for i, action in enumerate(Direction):
                    if not game.is_valid_action(action):
                        masked_q_values[0][i] = float('-inf')
                
                action_index = masked_q_values.argmax().item()
                return Direction(action_index)
    
    def replay(self):
        """
        Train the network on a batch of experiences from replay buffer.
        Optimized for speed and efficiency.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory - optimization using torch tensors directly
        batch = random.sample(self.memory, self.batch_size)
        
        # Use non_blocking=True to potentially overlap data transfer with computation
        states = torch.FloatTensor([e.state for e in batch]).to(device, non_blocking=True)
        actions = torch.LongTensor([e.action for e in batch]).to(device, non_blocking=True)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(device, non_blocking=True)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(device, non_blocking=True)
        dones = torch.BoolTensor([e.done for e in batch]).to(device, non_blocking=True)
        
        # Current Q values - more efficient batched operation
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network - Double DQN approach for better stability
        with torch.no_grad():
            # Get actions from main network for DDQN
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            # Get Q-values from target network for selected actions
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute Huber loss (more robust than MSE)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than setting to 0
        loss.backward()
        
        # Gradient clipping with slightly higher threshold
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=2.0)
        
        self.optimizer.step()
        
        # Update epsilon less frequently for speed
        if self.steps_done % 5 == 0 and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Store loss for analysis - only occasionally to save memory and time
        if self.update_count % 10 == 0:
            self.losses.append(loss.item())
        
        return loss.item()
    
    def save_model(self, filepath: str, metadata: Optional[dict] = None):
        """Save the trained model with comprehensive information"""
        save_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'steps_done': self.steps_done,
            'update_count': self.update_count,
            'losses': self.losses,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'target_update_frequency': self.target_update_frequency
            },
            'architecture': self.network_arch,  # Network architecture
            'device': str(device)
        }
        
        if metadata:
            save_data['metadata'] = metadata
        
        torch.save(save_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.epsilon_decay = checkpoint.get('epsilon_decay', self.epsilon_decay)
        self.epsilon_min = checkpoint.get('epsilon_min', self.epsilon_min)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.update_count = checkpoint.get('update_count', 0)
        self.losses = checkpoint.get('losses', [])
        
        print(f"Model loaded from {filepath}")
        
        # Print model information
        if 'hyperparameters' in checkpoint:
            print("Model hyperparameters:")
            for key, value in checkpoint['hyperparameters'].items():
                print(f"  {key}: {value}")
        
        if 'metadata' in checkpoint:
            print("Model metadata:")
            for key, value in checkpoint['metadata'].items():
                if key != 'training_results':  # Don't print large training data
                    print(f"  {key}: {value}")
    
    def get_model_info(self) -> dict:
        """Get comprehensive model information"""
        return {
            'name': self.name,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'batch_size': self.batch_size,
            'steps_done': self.steps_done,
            'update_count': self.update_count,
            'network_architecture': self.network_arch,
            'device': str(device),
            'training_losses': len(self.losses)
        }

def save_dqn_agent_for_viewing(agent: DQNAgent, filepath: str, metadata: Optional[dict] = None):
    """Save DQN agent in a simplified format for viewing (no training data)"""
    save_data = {
        'q_network_state_dict': agent.q_network.state_dict(),
        'hyperparameters': {
            'state_size': agent.state_size,
            'action_size': agent.action_size,
            'architecture': agent.network_arch
        },
        'metadata': metadata if metadata else {},
        'device': str(device)
    }
    
    torch.save(save_data, filepath)
    print(f"DQN agent saved for viewing to {filepath}")

def load_dqn_agent_for_viewing(filepath: str) -> DQNAgent:
    """Load DQN agent from simplified viewing format"""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Extract hyperparameters
    hyperparams = checkpoint['hyperparameters']
    
    # Create agent with inference-only settings
    agent = DQNAgent(
        state_size=hyperparams['state_size'],
        action_size=hyperparams['action_size'],
        epsilon=0.0,  # No exploration for viewing
        epsilon_min=0.0,
        memory_size=1  # Minimal memory for viewing
    )
    
    # Load the trained network weights
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['q_network_state_dict'])  # Same weights
    
    print(f"DQN agent loaded for viewing from {filepath}")
    
    # Print metadata if available
    if 'metadata' in checkpoint and checkpoint['metadata']:
        print("Agent metadata:")
        for key, value in checkpoint['metadata'].items():
            if key != 'training_results':  # Don't print large training data
                print(f"  {key}: {value}")
    
    return agent

def train_dqn_agent(episodes: int = 1000, width: int = 12, height: int = 12, 
                   render: bool = False, save_model_path: Optional[str] = "trained_models/best-dqn",
                   save_best: bool = True) -> Tuple[DQNAgent, List[dict]]:
    """
    Train DQN agent for specified number of episodes.
    """
    agent = DQNAgent(
        state_size=11,
        action_size=4,
        learning_rate=0.001,  # Slightly lower for more stable learning
        gamma=0.99,  # Higher discount factor for better long-term rewards
        epsilon=1.0,
        epsilon_decay=0.99,  # Faster epsilon decay to reduce exploration time
        epsilon_min=0.05,  # Slightly higher minimum epsilon for some exploration
        memory_size=100000,  # Reduced memory for faster access but still sufficient
        batch_size=128,  # Larger batch size for better parallelization on GPU
        target_update_frequency=100  # More frequent updates for faster convergence
    )
    
    results = []
    scores_window = deque(maxlen=100)  # For moving average
    best_score = 0
    best_model_state = None
    
    print(f"Training DQN Agent for {episodes} episodes...")
    print(f"Game size: {width}x{height}")
    
    for episode in range(episodes):
        game = SnakeGame(width, height, render=render)
        state = game.get_state_vector()
        
        total_reward = 0
        steps = 0
        episode_start_time = time.time()
        
        # Maximum steps calculation based on board size to avoid excessively long episodes
        # max_steps = min(width * height * 4, 1000)
        max_steps = min(width * height * 4, 1000)
        
        # Optimize experience collection by batching learning updates
        learn_every = 4  # Only learn every 4 steps to reduce computation
        
        while not game.game_over and steps < max_steps:
            # Get action from agent
            action = agent.get_action(state, game)
            
            # Execute action
            next_state_full, reward, done, info = game.step(action)
            next_state = game.get_state_vector()
            
            # Store experience
            agent.remember(state, action.value, reward, next_state, done)
            
            # Learn from experience less frequently for better performance
            if len(agent.memory) > agent.batch_size and steps % learn_every == 0:
                loss = agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            agent.steps_done += 1
            
            if render:
                game.render()
                time.sleep(0.01)
            
            if done:
                break
        
        episode_time = time.time() - episode_start_time
        scores_window.append(game.score)
        
        # Track best score and save best model state
        if game.score > best_score:
            best_score = game.score
            if save_best:
                best_model_state = {
                    'q_network_state_dict': agent.q_network.state_dict().copy(),
                    'target_network_state_dict': agent.target_network.state_dict().copy(),
                    'optimizer_state_dict': agent.optimizer.state_dict().copy(),
                    'episode': episode + 1,
                    'score': best_score
                }
        
        # Record episode results
        episode_result = {
            'episode': episode + 1,
            'score': game.score,
            'steps': steps,
            'total_reward': total_reward,
            'time': episode_time,
            'reason': info.get('reason', 'unknown') if 'info' in locals() else 'unknown',
            'efficiency': game.score / steps if steps > 0 else 0,
            'snake_length': len(game.snake_pos),
            'epsilon': agent.epsilon,
            'avg_score_100': np.mean(scores_window),
            'best_score': best_score
        }
        
        results.append(episode_result)
        
        # Print progress - less verbose for faster execution
        log_interval = min(100, max(10, episodes // 50))  # Adaptive logging based on total episodes
        if (episode + 1) % log_interval == 0:
            avg_score = np.mean(scores_window)
            print(f"Episode {episode + 1}/{episodes} | Avg Score: {avg_score:.2f} | Best: {best_score} | Epsilon: {agent.epsilon:.3f}")
            if agent.losses:
                print(f"  Memory: {len(agent.memory)} | Loss: {np.mean(agent.losses[-100:] if len(agent.losses) >= 100 else agent.losses):.4f}")
        
        game.close()
    
    # Restore best model if we saved it
    if save_best and best_model_state:
        agent.q_network.load_state_dict(best_model_state['q_network_state_dict'])
        agent.target_network.load_state_dict(best_model_state['target_network_state_dict'])
        agent.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
        print(f"Restored best model from episode {best_model_state['episode']} (score: {best_model_state['score']})")
    
    # Save model if path provided
    if save_model_path:
        training_metadata = {
            'training_episodes': episodes,
            'board_size': [width, height],
            'best_score': best_score,
            'final_avg_score': np.mean(scores_window),
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_training_time': sum(r['time'] for r in results)
        }
        agent.save_model(save_model_path, training_metadata)
        
        # Also save a simplified version for viewing
        save_dqn_agent_for_viewing(agent, "best_dqn_agent.pth", training_metadata)
        
        # Save the best model if save_model_path is provided
        if save_best and best_model_state and save_model_path:
            torch.save(best_model_state, save_model_path)
            print(f"Best model saved to {save_model_path}")
    
    return agent, results

def test_dqn_agent(agent: DQNAgent, episodes: int = 10, width: int = 12, height: int = 12, 
                  render: bool = True) -> List[dict]:
    """
    Test trained DQN agent with epsilon=0 (no exploration).
    """
    # Disable exploration for testing
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    results = []
    
    print(f"\nTesting DQN Agent for {episodes} episodes...")
    
    for episode in range(episodes):
        print(f"Test Episode {episode + 1}/{episodes}")
        
        game = SnakeGame(width, height, render=render)
        state = game.get_state_vector()
        
        total_reward = 0
        steps = 0
        episode_start_time = time.time()
        
        while not game.game_over and steps < 1000:
            # Get action from agent (no exploration)
            action = agent.get_action(state, game)
            
            # Execute action
            next_state_full, reward, done, info = game.step(action)
            next_state = game.get_state_vector()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render:
                game.render()
                time.sleep(0.05)  # Slow down for visibility
            
            if done:
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
        
        print(f"  Score: {game.score}, Steps: {steps}, Reward: {total_reward:.2f}")
        
        game.close()
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return results

def analyze_dqn_results(results: List[dict]):
    """Analyze and print statistics from DQN results"""
    if not results:
        return
    
    print("\n" + "="*50)
    print("DQN AGENT ANALYSIS")
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

def plot_training_progress(results: List[dict]):
    """Plot training progress"""
    try:
        episodes = [r['episode'] for r in results]
        scores = [r['score'] for r in results]
        avg_scores = [r.get('avg_score_100', 0) for r in results]
        epsilons = [r.get('epsilon', 0) for r in results]
        
        plt.figure(figsize=(15, 5))
        
        # Plot scores
        plt.subplot(1, 3, 1)
        plt.plot(episodes, scores, alpha=0.6, label='Score')
        plt.plot(episodes, avg_scores, label='Average Score (100 episodes)', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Training Progress - Scores')
        plt.legend()
        plt.grid(True)
        
        # Plot epsilon decay
        plt.subplot(1, 3, 2)
        plt.plot(episodes, epsilons, label='Epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (Epsilon)')
        plt.legend()
        plt.grid(True)
        
        # Plot efficiency
        plt.subplot(1, 3, 3)
        efficiencies = [r['efficiency'] for r in results]
        plt.plot(episodes, efficiencies, alpha=0.6, label='Efficiency')
        # Moving average
        if len(efficiencies) > 100:
            moving_avg = np.convolve(efficiencies, np.ones(100)/100, mode='valid')
            plt.plot(episodes[99:], moving_avg, label='Moving Average (100)', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Score/Steps')
        plt.title('Efficiency')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")

if __name__ == "__main__":
    # Quick test of DQN agent with optimized parameters
    print("Testing DQN Snake Agent with Optimized Parameters...")
    
    # Set torch to use optimized backend
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Train with optimized parameters
    agent, training_results = train_dqn_agent(
        episodes=5000,
        width=20, 
        height=20, 
        render=False, 
        save_best=True
    )
    
    # Test the trained agent
    test_results = test_dqn_agent(agent, episodes=3, width=20, height=20, render=True)
    
    # Analyze results
    analyze_dqn_results(test_results)
    
    # Plot training progress
    plot_training_progress(training_results)
    
    print("Optimized DQN Agent test completed!")