from torch import nn
import torch
from torch.distributions import Normal
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing device:{device}")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)     #输入state
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)    #输出动作分布的均值
        self.fc_std = nn.Linear(hidden_dim, action_dim)     #输出动作分布的标准差
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))      #relu激活函数，引入非线性
        x = self.relu(self.fc3(x)) 
        mean = self.tanh(self.fc_mean(x)) * 2   #Tanh激活函数，将均值限制在一定的范围内
        std = self.softplus(self.fc_std(x)) + 1e-3  #softplus激活函数，确保方差是正数

        return mean, std
    
    def select_action(self, s):
        with torch.no_grad():       #临时禁用梯度计算
            mu, sigma = self.forward(s)
            normal_dist = Normal(mu, sigma)
            action = normal_dist.sample()   #在正态分布中抽样
            # action = action.clamp(-2.0, 2.0)    #加限制
        
        return action





class Critic(nn.Module):        #输入state，来评估状态的alue
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)     #最终得到标量值
        self.relu = nn.ReLU()     #激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        value = self.fc4(x)     

        return value

class ReplayMemory:
    def __init__(self, batch_size):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.BATCH_SIZE = batch_size

    
    def add_memo(self, state, action, reward, value, done):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)

    def sample(self):
        num_state = len(self.state_cap)     #计算经验数据的数量 50
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)   #确定每一个批次的起始点
        memory_indicies = np.arange(num_state, dtype=np.int32)  #生成0到num_state的整数数组
        np.random.shuffle(memory_indicies)  #把整数组打乱
        batches = [memory_indicies[i:i+self.BATCH_SIZE]for i in batch_start_points]     #把打乱的memory_indi分成大小为BATCH_SIZE大小的几组数据

        return np.array(self.state_cap),\
            np.array(self.action_cap),\
            np.array(self.reward_cap),\
            np.array(self.value_cap),\
            np.array(self.done_cap),\
            batches
    
    def clear_memo(self):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []




class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size):
        self.LR_ACTOR = 3e-4
        self.LR_CRITIC = 3e-4
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.NUM_EPOCH = 10
        self.EPSILON_CLIP = 0.2

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LR_CRITIC)
        self.replay_buffer = ReplayMemory(batch_size)
        
    

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self. actor.select_action(state)
        value = self.critic.forward(state)
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]
        

    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict())     #复制当前actor网络的参数到old actor
        for epoch_i in range(self.NUM_EPOCH):       #让智能体对经验数据进行更新，多次使用，使策略网络和价值网络性能不断提升
            memo_states, memo_actions, memo_rewards, memo_values, memo_dones, batches = self.replay_buffer.sample()
            T = len(memo_rewards)
            memo_advantages = np.zeros(T, dtype=np.float32)

            for t in range(T):
                discount = 1 
                a_t = 0
                for k in range(t, T-1):
                    a_t += discount*(memo_rewards[k] + self.GAMMA * memo_values[k+1] * (1-int(memo_dones[k+1])) - memo_values[k])
                    discount *= self.GAMMA * self.LAMBDA
                memo_advantages[t] = a_t

            with torch.no_grad():
                memo_advantages_tensor = torch.tensor(memo_advantages).unsqueeze(1).to(device)
                memo_values_tensor = torch.tensor(memo_values).to(device)

            
            memo_states_tensor = torch.FloatTensor(memo_states).to(device)
            memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)

            for batch in batches:       #按批次对从回放缓冲区中采样得到的数据进行处理，计算 Actor 网络和 Critic 网络的损失，并通过反向传播更新网络参数
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(memo_states_tensor[batch])   #将当前的state输入到九网络得到均值和标准差
                    old_pi = Normal(old_mu, old_sigma)      #根据得出的均值和标准差创建正态分布对象old_pi
                batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch])    #求旧策略下，当前批次动作的对数概率

                mu, sigma = self.actor(memo_states_tensor[batch])
                pi = Normal(mu, sigma)
                batch_probs_tensor = pi.log_prob(memo_actions_tensor[batch])        #计算新策略下，当前批次动作的的对数概率

                ratio = torch.exp(batch_probs_tensor - batch_old_probs_tensor)      #
                surr1 = ratio * memo_advantages_tensor[batch]
                surr2 = torch.clamp(ratio, 1-self.EPSILON_CLIP, 1+self.EPSILON_CLIP) * memo_advantages_tensor[batch]

                actor_loss = -torch.min(surr1,surr2).mean()

                batch_returns = memo_advantages_tensor[batch] + memo_values_tensor[batch]   #优势函数+状态价值估计

                batch_old_values = self.critic(memo_states_tensor[batch])   #计算当前状态的critic状态估计值

                critic_loss = nn.MSELoss()(batch_old_values, batch_returns) #均方误差损失函数

                self.actor_optimizer.zero_grad()    #每次反向传播之前，将优化器中的梯度清零，以避免梯度累积。
                actor_loss.backward()       #对网络的损失进行反向传播，计算梯度
                self.actor_optimizer.step()     #根据计算到的梯度，更新网络的参数
                #self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LR_ACTOR)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_buffer.clear_memo()







    def save_policy(self):
        torch.save(self.actor.state_dict(), "ppo_polcy_pendulum_v1.para")

    
