import pygame
from cartpole import CartPole

def main():
    print("Initializing CartPole...")
    cartpole = CartPole()
    print("CartPole created, resetting...")
    cartpole.reset()
    print("Ready to start!")

    running = True
    action = 0  # 0 = left, 1 = right

    print("Controls:")
    print("LEFT ARROW or A: Push cart left")
    print("RIGHT ARROW or D: Push cart right")
    print("R: Reset episode")
    print("ESC or Q: Quit")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        cartpole.reset()
                        print("Episode reset!")

            # Get current key states for continuous control
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action = 0  # Push left
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action = 1  # Push right
            else:
                # No input - apply no force
                action = 2

            # Step the simulation
            state, reward, done = cartpole.step(action)

            # Render the current state
            cartpole.render()

            # Auto-reset if episode is done
            if done:
                print(f"Episode finished! Final state: {state}")
                pygame.time.wait(1000)  # Wait 1 second
                cartpole.reset()
                print("Episode auto-reset!")

    finally:
        cartpole.close()
        print("Game closed!")

if __name__ == "__main__":
    main()