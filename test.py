import pygame
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

# Snake block size
BLOCK_SIZE = 20

# Font and clock
SCORE_FONT = pygame.font.SysFont("comicsansms", 35)
CLOCK = pygame.time.Clock()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game AI Demo")

# Actions
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Model save path
MODEL_SAVE_PATH = "snake_dqn_model.pth"

class DeepQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return self.fc5(x)

class SnakeGameAI:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.x_change = 0
        self.y_change = 0
        self.snake_list = [[self.x, self.y]]
        self.snake_length = 1
        self.food_x = round(random.randrange(0, WIDTH - BLOCK_SIZE) / 20.0) * 20.0
        self.food_y = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / 20.0) * 20.0
        self.score = 0
        self.game_over = False
        return self.get_state()

    def step(self, action):
        if action == "LEFT" and self.x_change == 0:
            self.x_change = -BLOCK_SIZE
            self.y_change = 0
        elif action == "RIGHT" and self.x_change == 0:
            self.x_change = BLOCK_SIZE
            self.y_change = 0
        elif action == "UP" and self.y_change == 0:
            self.x_change = 0
            self.y_change = -BLOCK_SIZE
        elif action == "DOWN" and self.y_change == 0:
            self.x_change = 0
            self.y_change = BLOCK_SIZE

        self.x += self.x_change
        self.y += self.y_change

        if (self.x >= WIDTH or self.x < 0 or
            self.y >= HEIGHT or self.y < 0 or
            [self.x, self.y] in self.snake_list[:-1]):
            self.game_over = True
            return self.get_state(), -10, True

        snake_head = [self.x, self.y]
        self.snake_list.append(snake_head)
        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]

        if self.x == self.food_x and self.y == self.food_y:
            self.food_x = round(random.randrange(0, WIDTH - BLOCK_SIZE) / 20.0) * 20.0
            self.food_y = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / 20.0) * 20.0
            self.snake_length += 1
            self.score += 1
            return self.get_state(), 10, False

        return self.get_state(), 0, False

    def get_state(self):
        snake_head = self.snake_list[-1]
        left_danger = self.x - BLOCK_SIZE < 0 or [self.x - BLOCK_SIZE, self.y] in self.snake_list[:-1]
        right_danger = self.x + BLOCK_SIZE >= WIDTH or [self.x + BLOCK_SIZE, self.y] in self.snake_list[:-1]
        up_danger = self.y - BLOCK_SIZE < 0 or [self.x, self.y - BLOCK_SIZE] in self.snake_list[:-1]
        down_danger = self.y + BLOCK_SIZE >= HEIGHT or [self.x, self.y + BLOCK_SIZE] in self.snake_list[:-1]

        dir_l = self.x_change == -BLOCK_SIZE
        dir_r = self.x_change == BLOCK_SIZE
        dir_u = self.y_change == -BLOCK_SIZE
        dir_d = self.y_change == BLOCK_SIZE

        food_l = self.food_x < self.x
        food_r = self.food_x > self.x
        food_u = self.food_y < self.y
        food_d = self.food_y > self.y

        state = [
            int(left_danger), int(right_danger), int(up_danger), int(down_danger),
            int(dir_l), int(dir_r), int(dir_u), int(dir_d),
            int(food_l), int(food_r), int(food_u), int(food_d)
        ]

        return np.array(state, dtype=np.float32)

    def render(self):
        screen.fill(BLACK)
        pygame.draw.rect(screen, RED, [self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE])
        for block in self.snake_list:
            pygame.draw.rect(screen, GREEN, [block[0], block[1], BLOCK_SIZE, BLOCK_SIZE])
        value = SCORE_FONT.render("Score: " + str(self.score), True, BLUE)
        screen.blit(value, [10, 10])
        pygame.display.update()

def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeGameAI()
    state_size = len(env.get_state())
    action_size = len(ACTIONS)

    # Load the trained model
    model = DeepQNetwork(state_size, action_size).to(device)
    try:
        checkpoint = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(checkpoint['policy_net'])
        print("Loaded trained model successfully!")
    except FileNotFoundError:
        print("No trained model found. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    running = True
    while running:
        state = env.reset()
        done = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            # Get AI's action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_values = model(state_tensor)
                action_idx = action_values.argmax().item()
                action = ACTIONS[action_idx]

            # Execute action
            state, reward, done = env.step(action)
            env.render()
            CLOCK.tick(15)  # Control game speed

            if done:
                print(f"Game Over! Final Score: {env.score}")
                pygame.time.wait(1000)  # Wait a second before starting new game

    pygame.quit()

if __name__ == "__main__":
    run_demo()