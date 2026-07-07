import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory  #use in training
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

if torch.backendmps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



class Agent:
    def __init__(self,param_set):
        self.param_set = param_set

        with open("parameters.yaml","r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]


        self.alpha = params["alpha"]
        self.gamma = param_set["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.netword_sync_rate = params["network_sync_rate"]
        self.mini_batch_size = params["mini_batch_size"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None


    def run(self, is_training = True, render = False):
        env = gym.make("FlappyBird-v0",render_mode = "human" if render else None)
        
        num_states = env.observation_space.shape[0] #input dimension

        num_actions = env.action_space.n #output dimension

        policy_dqn = DQN(num_states,num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states,num_actions).to(device)

            #copy wt and bias vals from policy ==> target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(),lr=self.alpha)

        for episode in itertools.count():
            state,_ = env.reset()
            state = torch.tensor(state,dtype = torch.float,device = device)
            episode_rewards = 0
            terminated = False

            while not terminated:
                if is_training and random.random()<epsilon:
                    #Next action:
                    #(feed the observation to your agent here)
                    action = env.action_space.sample()
                    action = torch.tensor(action,dtype = torch.long,device = device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() #exploit

                #Processing: terminate => done
                next_state,reward,terminated , _, _ = env.step(action.item())


                #create tensors
                reward = torch.tensor(reward,dtype=torch.float,device = device)
                next_state = torch.tensor(next_state,dtype = torch.float,device = device)
    

                if is_training:
                    memory.append((state,action,new_state,reward,terminated))
                    steps += 1
                
                state = new_state
                episode_rewards += reward
            
            print(f"episode = {episode+1} with total reward = {episode_rewards} & epsilon{epsilon}")


            if is_training:
                #epsilon decay
                epsilon =  max(epsilon * self.epsilon_decay , self.epsilon.min)

            if is_training and len(memory) > self.mini_batch_size: #sync networks
                #get sample 
                mini_batch = memory.sample(self.mini_batch_size)

                optimize(mini_batch,policy_dqn,target_dqn)

                #sync the network
                if steps > self.netword_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0
            #env.close
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        #get experience ==> in batch for fast training
        state,action,next_state,reward,terminated = zip(*mini_batch)

        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        rewards = torch.stack(rewards)
        terminated = torch.tensor(terminated).float().to(device)


        

    
        with torch.no_grad():
            target_q = reward + (1-terminated) * self.gamma * target_dqn(next_state).max(dim=1)[0]

        current_q = policy_dqn(state).gather(dim=1,index = action.unsqueeze(dim=1)).squeeze()


            #loss
        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

