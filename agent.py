import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory  #use in training

if torch.backendmps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else
    device = "cpu"


def run(self, is_training = True, render = False)
    env = gym.make("FlappyBird-v0",render_mode = "human" if render else None)
    
    num_states = env.observation_space.shape[0] #input dimension

    num_actions = env.action_space.n #output dimension

    policy_dqn = DQN(num_states,num_actions).to(device)

    if is_training:
        memory = ReplayMemory(self.replay_memory_size)
        epsilon = self.epsilon_init

    for episode in itertools.count():
        state,_ = env.reset()
        episode_rewards = 0
        terminated = False

        while not terminated:
            if is_training and random.random()<epsilon:
                #Next action:
                #(feed the observation to your agent here)
                action = env.action_space.sample()
            else:
                action = policy_dqn(state).argmax() #exploit

            #Processing: terminate => done
            next_state,reward,terminated , _, _ = env.step(action)

   

            if is_training:
                memory.append((state,action,new_state,reward,terminated))
            
            state = new_state
            episode_rewards += reward
        
        print(f"episode = {episode+1} with total reward = {episode_rewards} & epsilon{epsilon}")

        #epsilon decay
        epsilon -  max(epsilon * self.epsilon_decay , self.epsilon.min)
        #env.close