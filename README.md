# Snake AI DQN

A Python implementation of the classic Snake game powered by Deep Q-Learning Network (DQN). This project combines PyGame for visualization with PyTorch for deep reinforcement learning, creating an AI agent that learns to play Snake through experience.

<p align="center"> 


https://github.com/user-attachments/assets/09819538-3a64-4d71-b6bd-bed7e211ccbf


</p>

## Features

- Classic Snake game implementation using PyGame
- Deep Q-Learning Network (DQN) with PyTorch
- Experience replay for stable training
- Model save/load functionality
- Real-time visualization of training
- Configurable hyperparameters
- Support for both CPU and CUDA training

## Requirements

```
pygame
numpy
torch
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abhigyan126/Snake-DQN
cd Snake-DQN
```

2. Install dependencies:
```bash
pip install pygame numpy torch
```

## Usage

Run the main script to start training:

```bash
python train.py
```

You'll be presented with three options:
1. Start new training
2. Continue training existing model
3. Exit

The training process will display:
- Real-time game visualization
- Episode progress
- Total rewards
- Current exploration rate (epsilon)

## Technical Details

### Neural Network Architecture
- Input layer: 12 nodes (state space)
- Hidden layers: 64 → 64 → 128 → 64 nodes
- Output layer: 4 nodes (action space)
- Activation function: Leaky ReLU

### State Representation
The game state consists of 12 binary values:
- Danger detection (4 directions)
- Current direction (4 possibilities)
- Food location relative to snake (4 directions)

### Training Parameters
- Replay buffer size: 10,000
- Batch size: 64
- Learning rate: 0.001
- Discount factor (gamma): 0.99
- Initial epsilon: 1.0
- Minimum epsilon: 0.01
- Epsilon decay: 0.995

## Project Structure

```
snake-ai-dqn/
├── train.py              # Main game and training logic
├── test.py               # Test scrip to run the model
├── README.md             # Project documentation
└── snake_dqn_model.pth   # Saved model checkpoints
```

