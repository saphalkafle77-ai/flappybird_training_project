import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN

if torch.backendmps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else
    device = "cpu"

     
env = gym.make("FlappyBird-v0",render_mode = "human")

state,_ = env.reset()
while True:
    #Next action:
    #(feed the observation to your agent here)
    action = env.action_space.sample()

    #Processing: terminate => done
    next_state,reward,terminated,_ = env.step(action)

    #checking if the player is still alive
    if terminated:
        break
env.close