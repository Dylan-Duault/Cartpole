import pygame
import numpy as np
from cartpole import CartPole
from dqn import DQNAgent
import time

def evaluate_agent(model_path='dqn_model_final.pth', episodes=5, render=True, delay=0.02):
    """
    Evaluate a trained DQN agent

    Args:
        model_path: Path to the saved model
        episodes: Number of episodes to evaluate
        render: Whether to show the visual rendering
        delay: Delay between steps (for human observation)
    """
    env = CartPole()
    agent = DQNAgent(state_size=4, action_size=3)

    # Load the trained model
    try:
        agent.load(model_path)
        agent.epsilon = 0  # No exploration during evaluation
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return

    scores = []

    print(f"Evaluating agent for {episodes} episodes...")
    print("-" * 50)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        if render:
            print(f"\nEpisode {episode + 1}")
            print("Controls:")
            print("  ESC: Stop evaluation")
            print("  SPACE: Pause/unpause")
            print("  LEFT/A: Override AI with push left")
            print("  RIGHT/D: Override AI with push right")
            print("  AI chooses action, your input overrides when pressed")

        running = True
        paused = False

        while running:
            if render:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        env.close()
                        return scores
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        elif event.key == pygame.K_SPACE:
                            paused = not paused

                if paused:
                    env.render()
                    continue

            # AI chooses action first
            ai_action = agent.act(state, training=False)

            # Check for manual override
            if render:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    action = 0  # Manual override: push left
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    action = 1  # Manual override: push right
                else:
                    action = ai_action  # Use AI's choice
            else:
                action = ai_action

            # Take action
            next_state, reward, done = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(delay)

            if done:
                break

        scores.append(total_reward)
        print(f"Episode {episode + 1}: Score = {total_reward:.1f}, Steps = {steps}")

    env.close()

    # Print statistics
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({episodes} episodes):")
    print(f"Average Score: {avg_score:.1f} ± {std_score:.1f}")
    print(f"Best Score: {max(scores):.1f}")
    print(f"Worst Score: {min(scores):.1f}")

    if avg_score >= 195:
        print("🎉 Agent has solved the CartPole environment!")
    else:
        print(f"Agent needs improvement. Target: 195, Current: {avg_score:.1f}")

    return scores

def evaluate_agent_headless(model_path='dqn_model_final.pth', episodes=100):
    """
    Evaluate agent without rendering (for quick testing)
    """
    env = CartPole()
    agent = DQNAgent(state_size=4, action_size=3)

    try:
        agent.load(model_path)
        agent.epsilon = 0
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return []

    scores = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state, training=False)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

    env.close()
    return scores

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    parser.add_argument('--model', default='dqn_model_final.pth',
                       help='Path to model file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to evaluate')
    parser.add_argument('--headless', action='store_true',
                       help='Run without rendering')
    parser.add_argument('--delay', type=float, default=0.02,
                       help='Delay between steps (seconds)')

    args = parser.parse_args()

    if args.headless:
        scores = evaluate_agent_headless(args.model, args.episodes)
        if scores:
            print(f"Average score over {args.episodes} episodes: {np.mean(scores):.1f}")
    else:
        evaluate_agent(args.model, args.episodes, render=True, delay=args.delay)