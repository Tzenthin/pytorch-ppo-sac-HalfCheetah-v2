import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.autograd import Variable
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        torch.nn.init.constant_(m.bias, 0)

def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 *\
        np.log(2 * np.pi) - log_std    # num_env * frames * act_size
    #print(log_density.shape)
    #print(log_density)
    log_density = log_density.sum(dim=1, keepdim=True) # num_env * frames * 1
    #print(log_density.shape)
    #assert 0==1
    return log_density

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(ActorCritic, self).__init__()

        self.logstd = nn.Parameter(torch.zeros(num_actions))

        self.a_fc1 = nn.Linear(num_inputs, hidden_dim)
        self.a_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.a_fc3 = nn.Linear(hidden_dim, num_actions)
        self.a_fc3.weight.data.mul_(0.1)
        self.a_fc3.bias.data.mul_(0.0)

        self.c_fc1 = nn.Linear(num_inputs, hidden_dim)
        self.c_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.c_fc3 = nn.Linear(hidden_dim, 1)
        self.c_fc3.weight.data.mul_(0.1)
        self.c_fc3.bias.data.mul_(0.0)

        #self.apply(weights_init_)

    def forward(self, x):
        a = F.tanh(self.a_fc1(x))
        a = F.tanh(self.a_fc2(a))
        mean = self.a_fc3(a)
        #mean = F.tanh(self.a_fc3(a))
        

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        #logprob = log_normal_density(action, mean, logstd, std)

        v = F.tanh(self.c_fc1(x))
        v = F.tanh(self.c_fc2(v))
        #v = F.relu(self.c_fc3(v))
        v = self.c_fc3(v)

        return v, action, mean
    
    def evaluate(self, x, action):
        v, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        #evaluate
        logprob = log_normal_density(action, mean, logstd, std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()

        return v, logprob, dist_entropy



def cal_gae(reward, mask, gamma, lam, values):
    #reward = torch.Tensor(reward)
    #mask = torch.Tensor(mask)
    #print(reward.shape, mask.shape, values.shape)
    #print(values)
    values_t = values.detach().squeeze().cpu()
    mask = mask.detach().squeeze().cpu()
    reward = reward.detach().squeeze().cpu()

    #print(values_t.shape, mask.shape, reward.shape)
    num_steps = reward.shape[0]

    targets = torch.zeros_like(reward)
    advants = torch.zeros_like(reward)
    #gae = torch.zeros_like(reward)
    
    running_target = 0
    running_advants = 0
    previous_value = 0
    
    for t in reversed(range(0,num_steps)):
        running_target = reward[t] + mask[t]*gamma*running_target
        delta = reward[t] + mask[t]*gamma*previous_value - values_t[t]
        #gae = delta + mask[t]*gamma*lam*gae
        running_advants = delta + mask[t]*gamma*lam*running_advants

        targets[t] = running_target
        previous_value = values_t[t]
        advants[t] = running_advants

    advants = (advants - advants.mean())/advants.std()

    return targets, advants

def ppo_update(policy, memory, optimizer, K_epochs, batch_size, gamma, lam, clip, coeff_entropy):
    #print(len(memory))
    #policy = policy.to(device)
    memory = np.asarray(memory)
    '''
    print(len(memory))
    print(memory.shape)
    print(memory[:,0].shape)
    print(np.squeeze(memory[:, 3]).shape)
    print(torch.FloatTensor(np.squeeze(memory[:,0]).shape))
    assert 0==1
    '''

    #print()
    state = np.vstack(memory[:, 0])#torch.from_numpy(memory[:,0])
    #print(state)
    #print(state.shape)
    state = torch.FloatTensor(state).to(device)
    #print(state.shape)
    #assert 0==1
    action = np.vstack(memory[:,1])
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(list(memory[:,2])).to(device)
    mask = torch.FloatTensor(list(memory[:,3])).to(device)
    #print(state.shape, action.shape, reward.shape, mask.shape)
    #print(mask)
    #assert 0==1
    values,  _, _ = policy(state)
    targets ,advantages = cal_gae(reward, mask, gamma, lam, values)


    old_values, old_logprob, _ = policy.evaluate(state, action)

    
    #print(old_logprob.shape, old_logprob)
    #print(old_logprob.shape)
    #assert 0==1
    #print(old_values.shape)
    #old_values = old_values.detach().squeeze().cpu().numpy()
    #print(old_values.shape)
    #old_logprob = old_logprob.
    #assert 0==1
    
    #print(mask)
    #print(targets.shape, advantages.shape, advantages)
    #assert 0==1

    n = len(state)
    arr = np.arange(n)
    #print(state)
    #print(state[0:3])
    #assert 0==1
    #batch_index = arr[batch_size*i : batch_size*(i+1)]

    for epoch in range(K_epochs):
        np.random.shuffle(arr)
        for i in range(n//batch_size):
            batch_index = arr[batch_size*i : batch_size*(i+1)]
            batch_index = torch.LongTensor(batch_index)
            inputs_samples = state[batch_index]#torch.Tensor(state)[batch_index]
            action_samples = action[batch_index]#torch.Tensor(action)[batch_index]
            targets_samples = targets[batch_index].to(device).unsqueeze(1)#torch.Tensor(targets)[batch_index]
            advantages_samples = advantages[batch_index].to(device).unsqueeze(1)#torch.Tensor(advantages)[batch_index]
            logprob_sample = old_logprob[batch_index].detach().to(device)#.squeeze()
            values_sample = old_values[batch_index].detach().to(device)
            #print(values_sample.shape)

            new_v, new_logprob, dist_entropy = policy.evaluate(inputs_samples, action_samples)
            #print(logprob_sample.shape, new_logprob.shape)
            ratio = torch.exp(new_logprob - logprob_sample)#.squeeze()
            #print(ratio.shape, advantages_samples.shape)

            clipped_ratio = torch.clamp(ratio, 1.0-clip, 1.0+clip)
            #print(clipped_ratio.shape)
            surrogate1 = ratio * advantages_samples
            surrogate2 = clipped_ratio * advantages_samples
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            #print('new v ',new_v, '  sample v',values_sample)
            #print('target', targets_samples)
            clipped_values = values_sample + torch.clamp(new_v - values_sample, -0.5, 0.5)
            c_loss1 = F.mse_loss(clipped_values, targets_samples)
            c_loss2 = F.mse_loss(new_v, targets_samples)
            critic_loss = torch.max(c_loss1, c_loss2).mean()
            #print('loss    ',c_loss1, c_loss2)
                        
            #critic_loss = F.mse_loss(new_v, targets_samples)
            #print(targets_samples.shape, new_v.shape)
            
            #print(policy_loss, critic_loss, dist_entropy)
            loss = policy_loss + 0.5*critic_loss #- coeff_entropy*dist_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




            
            
            







'''
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
'''


'''
class ActorCritic(nn.Module):
    def __init__(self,num_inputs, num_actions, hidden_dim): # state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        self.logstd = nn.Parameter(torch.zeros(num_actions))

        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(num_inputs, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
                #nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(num_inputs, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
                )
        
    def forward(self):
        raise NotImplementedError
       
    def act(self, state, memory):
        action_mean = self.actor(state)
        logstd = self.logstd.expand_as(action_mean)
        std = torch.exp(logstd)
        normal = Normal(action_mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        action_logprob = normal.log_prob(x_t)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action, F.tanh(action_mean)

    def evaluate(self, state, action):
        state_value = self.critic(state)
        action_mean = torch.squeeze(self.actor(state))
        #print(action_mean.shape)
        log_std = self.logstd.expand_as(action_mean)
        #print(self.logstd.shape)
        #print(log_std.shape)
        std = torch.exp(log_std)
        normal = Normal(action_mean, std)
        action_logprob = normal.log_prob(torch.squeeze(action))#.mean(-1)
        #print('action_logprobs',action_logprob.shape)
        dist_entropy = normal.entropy().mean(-1)
        #print('dist_entropy    ',dist_entropy.shape)
        return action_logprob, torch.squeeze(state_value), dist_entropy

'''


'''

class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        print(self.policy)
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        #print(state)
        #print(memory)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, mean = self.policy_old.act(state, memory)
        action = action.cpu().data.numpy().flatten()
        mean = mean.cpu().data.numpy().flatten()
        #print(action, mean)
        #assert 0==1
        #print(action, action.flatten())
        #assert 0==1
        return action, mean
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        #print('rewards.shape',len(rewards))
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        #print('old_logprobs',old_logprobs.shape)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp( (logprobs - old_logprobs.detach()).mean(-1)   )
            #print('ratio', ratios, ratios.shape)

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
'''





'''
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = torch.squeeze(self.actor(state))
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
'''

