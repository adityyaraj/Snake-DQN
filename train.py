import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import threading
import queue

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
FONT = pygame.font.SysFont("bahnschrift", 25)
SCORE_FONT = pygame.font.SysFont("comicsansms", 35)
CLOCK = pygame.time.Clock()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game with Deep Q-Learning")

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


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


class DQNAgent:
    def __init__(self, state_size, action_size, device, load_model=False):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.policy_net = DeepQNetwork(state_size, action_size).to(device)
        self.target_net = DeepQNetwork(state_size, action_size).to(device)

        if load_model and os.path.exists(MODEL_SAVE_PATH):
            checkpoint = torch.load(MODEL_SAVE_PATH)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = 1.0

        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon
        }, MODEL_SAVE_PATH)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_values = self.policy_net(state_tensor)
            return action_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_snake(load_existing=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeGameAI()
    state_size = len(env.get_state())
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size, device, load_model=load_existing)
    episodes = 2000
    update_target_frequency = 100

    def game_render_thread():
        while True:
            env.render()
            CLOCK.tick(15)

    render_thread = threading.Thread(target=game_render_thread, daemon=True)
    render_thread.start()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.select_action(state)
            action = ACTIONS[action_idx]
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.memory.push(state, action_idx, reward, next_state, done)
            agent.train()
            state = next_state

        if episode % update_target_frequency == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")

        if (episode + 1) % 25 == 0:
            agent.save_model()

    agent.save_model()
    pygame.quit()


def main():
    print("\nChoose an option:")
    print("1. Start new training")
    print("2. Continue training existing model")
    print("3. Exit")

    choice = int(input("Enter your choice (1-3): "))
    if choice == 1:
        train_snake(load_existing=False)
    elif choice == 2:
        train_snake(load_existing=True)
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()