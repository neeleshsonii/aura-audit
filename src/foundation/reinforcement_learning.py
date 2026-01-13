import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# SEED FOR REPRODUCIBILITY
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class CustomerSupportQLearning:
    """
    Task 2.2: Implement Q-Learning for optimizing customer support routing decisions.
    
    States: Combination of (intent, sentiment)
    Actions: [escalate_to_human, use_automated_response, request_more_info]
    Rewards: Based on customer satisfaction and efficiency
    """
    
    def __init__(self, n_states=15, n_actions=3, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1):
        self.n_states = n_states  # 5 intents * 3 sentiments
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Action names for interpretability
        self.action_names = [
            'escalate_to_human',
            'use_automated_response',
            'request_more_info'
        ]
    
    def get_state_id(self, intent, sentiment):
        """Convert intent and sentiment to state ID."""
        # Map intent (0-4) and sentiment (0-2) to single state ID
        return intent * 3 + sentiment
    
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def get_reward(self, intent, sentiment, action):
        """
        Define reward function based on domain knowledge.
        """
        # Sentiment: 0=Satisfied, 1=Neutral, 2=Frustrated
        # Actions: 0=escalate, 1=automated, 2=more_info
        
        # Frustrated customers
        if sentiment == 2:
            if action == 0:  # Escalate
                return 10
            elif action == 1:  # Automated
                return -10
            else:  # More info
                return 5
        
        # Satisfied/Neutral with simple issues
        elif intent in [2, 4]:  # login_issue, feature_request (simpler)
            if action == 1:  # Automated
                return 8
            elif action == 0:  # Escalate
                return -5
            else:
                return 3
        
        # Complex issues (billing, refund)
        elif intent in [0, 3]:  # billing, refund_request
            if action == 0:  # Escalate
                return 9
            elif action == 1:  # Automated
                return -3
            else:
                return 6
        
        # Tech support (medium complexity)
        else:
            if action == 2:  # More info first
                return 7
            elif action == 0:  # Or escalate
                return 5
            else:
                return 2
    
    def update(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update rule
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q
    
    def train(self, episodes=1000, log_interval=100):
        """
        Train the Q-learning agent.
        
        Simulates customer support scenarios with different intents and sentiments.
        """
        rewards_history = []
        
        for episode in range(episodes):
            # Random starting state (intent, sentiment)
            intent = np.random.randint(0, 5)
            sentiment = np.random.randint(0, 3)
            state = self.get_state_id(intent, sentiment)
            
            episode_reward = 0
            # Simulate a customer support interaction (3-5 steps)
            steps = np.random.randint(3, 6)
            for step in range(steps):
                # Choose action
                action = self.choose_action(state)
                
                # Get reward
                reward = self.get_reward(intent, sentiment, action)
                episode_reward += reward
                
                # Simulate next state (could be new intent or resolved)
                if np.random.random() < 0.3:  # 30% chance state changes
                    intent = np.random.randint(0, 5)
                    sentiment = max(0, sentiment - 1)  # Sentiment improves slightly
                
                next_state = self.get_state_id(intent, sentiment)
                # Update Q-table
                self.update(state, action, reward, next_state)
                
                state = next_state
            rewards_history.append(episode_reward)
            
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(rewards_history[-log_interval:])
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
        
        return rewards_history
    
    def get_policy(self):
        """Extract the learned policy from Q-table."""
        policy = np.argmax(self.q_table, axis=1)
        return policy
    
    def visualize_policy(self):
        """Visualize the learned policy."""
        policy = self.get_policy()
        
        # Create a grid representation
        intents = ['Billing', 'Tech', 'Login', 'Refund', 'Feature']
        sentiments = ['Satisfied', 'Neutral', 'Frustrated']
        
        policy_matrix = policy.reshape(5, 3)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(policy_matrix, cmap='viridis', aspect='auto')
        
        # Add labels
        for i in range(5):
            for j in range(3):
                action_id = policy_matrix[i, j]
                action_name = self.action_names[action_id].replace('_', '\n')
                plt.text(j, i, action_name, ha='center', va='center',
                        color='white', fontsize=9, weight='bold')
        
        plt.xticks(range(3), sentiments)
        plt.yticks(range(5), intents)
        plt.xlabel('Sentiment')
        plt.ylabel('Intent')
        plt.title('Learned Q-Learning Policy for Customer Support Routing')
        plt.colorbar(label='Action ID')
        
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/q_learning_policy.png', dpi=150, bbox_inches='tight')
        print("\nPolicy visualization saved to outputs/q_learning_policy.png")
        plt.close()
    
    def visualize_training(self, rewards_history):
        """Plot training rewards over time."""
        plt.figure(figsize=(10, 6))
        # Moving average
        window = 50
        moving_avg = np.convolve(rewards_history, 
                                 np.ones(window)/window, mode='valid')
        plt.plot(rewards_history, alpha=0.3, label='Episode Reward')
        plt.plot(range(window-1, len(rewards_history)), moving_avg, 
                label=f'{window}-Episode Moving Average', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Q-Learning Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('outputs/q_learning_training.png', dpi=150, bbox_inches='tight')
        print("Training progress saved to outputs/q_learning_training.png")
        plt.close()

def run_reinforcement_learning():
    """
    Task 2.2: Main function to run Q-Learning experiment.
    """
    print("\n" + "="*60)
    print("TASK 2.2: REINFORCEMENT LEARNING (Q-Learning)")
    print("="*60)
    
    # Initialize Q-Learning agent
    agent = CustomerSupportQLearning(
        n_states=15,  # 5 intents * 3 sentiments
        n_actions=3,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    print("\nTraining Q-Learning agent...")
    print("Simulating customer support routing scenarios...")
    
    # Train the agent
    rewards_history = agent.train(episodes=1000, log_interval=200)
    
    # Visualize results
    print("\nGenerating visualizations...")
    agent.visualize_training(rewards_history)
    agent.visualize_policy()
    
    # Display final policy
    print("\n=== Learned Policy ===")
    policy = agent.get_policy()
    intents = ['Billing', 'Tech', 'Login', 'Refund', 'Feature']
    sentiments = ['Satisfied', 'Neutral', 'Frustrated']
    
    print("\nOptimal Actions for each (Intent, Sentiment) combination:")
    print("-" * 70)
    for i, intent in enumerate(intents):
        for j, sentiment in enumerate(sentiments):
            state_id = i * 3 + j
            action_id = policy[state_id]
            action_name = agent.action_names[action_id]
            print(f"{intent:15} + {sentiment:12} -> {action_name}")
    
    # Save Q-table
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(agent, 'models/q_learning_agent.pkl')
    print("\nQ-Learning agent saved to models/q_learning_agent.pkl")
    
    return agent

if __name__ == "__main__":
    agent = run_reinforcement_learning()
