import pygame
import numpy as np
import math

class CartPole:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

        # Physics constants (simplified for AI training)
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5  # half the pole length
        self.force_magnitude = 10.0
        self.dt = 0.02  # time step

        # State variables
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = 0.0  # radians from vertical
        self.pole_angular_velocity = 0.0

        # Boundaries
        self.x_threshold = 2.4
        self.angle_threshold = math.pi / 3  # 60 degrees

        # Pygame setup
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Cartpole")
        self.clock = pygame.time.Clock()

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)

        # Rendering scale
        self.scale = 100  # pixels per meter

    def reset(self):
        """Reset the environment to initial state"""
        self.cart_position = np.random.uniform(-1.0, 1.0)
        self.cart_velocity = np.random.uniform(-2.0, 2.0)
        self.pole_angle = np.random.uniform(-0.30, 0.30)
        self.pole_angular_velocity = np.random.uniform(-0.05, 0.05)
        return self.get_state()

    def get_state(self):
        """Return current state as numpy array"""
        return np.array([
            self.cart_position,
            self.cart_velocity,
            self.pole_angle,
            self.pole_angular_velocity
        ])

    def is_done(self):
        """Check if episode is terminated"""
        return (abs(self.cart_position) > self.x_threshold or
                abs(self.pole_angle) > self.angle_threshold)

    def step(self, action):
        """
        Apply action and update physics
        action: 0 = push left, 1 = push right, 2 = no action
        """
        # Apply force based on action
        if action == 1:
            force = self.force_magnitude
        elif action == 0:
            force = -self.force_magnitude
        else:  # action == 2 or any other value means no force
            force = 0.0

        # Simplified physics calculations
        costheta = math.cos(self.pole_angle)
        sintheta = math.sin(self.pole_angle)

        # Total mass
        total_mass = self.cart_mass + self.pole_mass

        # Pole physics
        temp = (force + self.pole_mass * self.pole_length *
                self.pole_angular_velocity**2 * sintheta) / total_mass

        pole_acc = (self.gravity * sintheta - costheta * temp) / \
                   (self.pole_length * (4.0/3.0 - self.pole_mass * costheta**2 / total_mass))

        cart_acc = temp - self.pole_mass * self.pole_length * pole_acc * costheta / total_mass

        # Update velocities
        self.cart_velocity += cart_acc * self.dt
        self.pole_angular_velocity += pole_acc * self.dt

        # Update positions
        self.cart_position += self.cart_velocity * self.dt
        self.pole_angle += self.pole_angular_velocity * self.dt

        # Calculate reward (simple: +1 for staying upright)
        reward = 1.0 if not self.is_done() else 0.0

        return self.get_state(), reward, self.is_done()

    def render(self):
        """Render the cartpole using pygame"""
        self.screen.fill(self.WHITE)

        # Calculate screen coordinates
        cart_x = self.width // 2 + int(self.cart_position * self.scale)
        cart_y = self.height // 2 + 50

        # Draw track
        pygame.draw.line(self.screen, self.BLACK,
                        (50, cart_y + 20), (self.width - 50, cart_y + 20), 3)

        # Draw cart (rectangle)
        cart_width, cart_height = 50, 30
        cart_rect = pygame.Rect(cart_x - cart_width//2, cart_y - cart_height//2,
                               cart_width, cart_height)
        pygame.draw.rect(self.screen, self.BLUE, cart_rect)

        # Draw pole
        pole_end_x = cart_x + int(self.pole_length * self.scale * 2 * math.sin(self.pole_angle))
        pole_end_y = cart_y - int(self.pole_length * self.scale * 2 * math.cos(self.pole_angle))

        pygame.draw.line(self.screen, self.RED, (cart_x, cart_y),
                        (pole_end_x, pole_end_y), 5)

        # Draw pole tip
        pygame.draw.circle(self.screen, self.GREEN, (pole_end_x, pole_end_y), 8)

        # Display state information
        font = pygame.font.Font(None, 36)
        state = self.get_state()
        text_lines = [
            f"Cart Pos: {state[0]:.3f}",
            f"Cart Vel: {state[1]:.3f}",
            f"Pole Angle: {state[2]:.3f}",
            f"Pole Vel: {state[3]:.3f}"
        ]

        for i, line in enumerate(text_lines):
            text = font.render(line, True, self.BLACK)
            self.screen.blit(text, (10, 10 + i * 30))

        # Show if episode is done
        if self.is_done():
            done_text = font.render("EPISODE DONE!", True, self.RED)
            self.screen.blit(done_text, (self.width//2 - 100, 50))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        """Clean up pygame"""
        pygame.quit()