import numpy as np
import matplotlib.pyplot as plt
from cartpole import CartPole
from dqn import DQNAgent
import torch

def train_dqn(episodes=1000, max_steps=500, target_update=10, save_interval=100):
    env = CartPole()
    agent = DQNAgent(state_size=4, action_size=3)

    scores = []
    losses = []
    avg_scores = []

    print(f"Training DQN on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Episodes: {episodes}, Max steps per episode: {max_steps}")
    print("-" * 50)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Train the agent
            if len(agent.memory) > 32:
                loss = agent.replay()
                episode_losses.append(loss)

            if done:
                break

        scores.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0)

        # Update target network
        if episode % target_update == 0:
            agent.update_target_network()

        # Calculate running average
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        else:
            avg_scores.append(np.mean(scores))

        # Print progress
        if episode % 50 == 0:
            avg_score = avg_scores[-1]
            print(f"Episode {episode:4d} | Score: {total_reward:6.1f} | "
                  f"Avg Score: {avg_score:6.1f} | Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {losses[-1]:.6f}")

        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            agent.save(f'dqn_model_episode_{episode}.pth')
            print(f"Model saved at episode {episode}")

        # Early stopping if solved
        if len(avg_scores) >= 100 and avg_scores[-1] >= 195:
            print(f"\nSolved in {episode} episodes! Average score: {avg_scores[-1]:.1f}")
            break

    # Save final model
    agent.save('dqn_model_final.pth')
    print("Final model saved as 'dqn_model_final.pth'")

    env.close()

    return scores, losses, avg_scores, agent

def plot_training_results(scores, losses, avg_scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot scores
    ax1.plot(scores, alpha=0.6, color='blue', label='Episode Score')
    ax1.plot(avg_scores, color='red', linewidth=2, label='Average Score (100 episodes)')
    ax1.axhline(y=195, color='green', linestyle='--', label='Solved Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.legend()
    ax1.grid(True)

    # Plot losses
    ax2.plot(losses, color='orange', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train the agent
    scores, losses, avg_scores, trained_agent = train_dqn(
        episodes=1000,
        max_steps=500,
        target_update=10,
        save_interval=100
    )

    # Plot results
    plot_training_results(scores, losses, avg_scores)

    print("\nTraining completed!")
    print(f"Final average score: {avg_scores[-1]:.1f}")
    print(f"Final epsilon: {trained_agent.epsilon:.3f}")