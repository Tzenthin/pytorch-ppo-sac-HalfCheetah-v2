import argparse
import datetime
import gym
import numpy as np

import os
import logging
import sys
import socket
import torch

from ppo_model import Memory, PPO, ActorCritic

'''
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')
'''



hostname = socket.gethostname()
if not os.path.exists('./ppo_log/' + hostname):
    os.makedirs('./ppo_log/' + hostname)
output_file = './ppo_log/' + hostname + '/output.log'
cal_file = './ppo_log/' + hostname + '/cal.log'
logger = logging.getLogger('mylogger')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(output_file, mode='a')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
logger_cal = logging.getLogger('loggercal')
logger_cal.setLevel(logging.INFO)
cal_f_handler = logging.FileHandler(cal_file, mode='a')
file_handler.setLevel(logging.INFO)
logger_cal.addHandler(cal_f_handler)




#env = gym.make(args.env_name)
'''
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
'''



def main():
    ############## Hyperparameters ##############
    env_name = "HalfCheetah-v2"
    render = True#False
    #solved_reward = 300         # stop training if avg_reward > solved_reward
    #log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 4000        # max timesteps in one episode
    
    update_timestep = 1000      # update policy every n timesteps
    #action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 4               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    hidden_dim = 256
    random_seed = None
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print('state_dim',state_dim, '   action_dim',action_dim)
    print('load env succeed!*********************************')



    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, hidden_dim, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy.load_state_dict(torch.load('ppo_model/ppo_500.pth'))
    print(lr,betas)
    
    # logging variables
    episode_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action, mean = ppo.select_action(state, memory)
            next_state, reward, done, _ = env.step(action)
            # Saving reward:
            '''
            memory.rewards.append(reward)
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                print('update')
                memory.clear_memory()
                time_step = 0
            episode_reward += reward
            '''
            state = next_state
            if render:
                env.render()
            if done:
                break
        
        #avg_length += t
        '''
        logger.info("Episode: {}, total numsteps: {}, reward: {}".format(i_episode, time_step, round(episode_reward, 2)))
        #logger.info('Env %d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f,%s' % (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, result))
        logger_cal.info(episode_reward)

        # save every 500 episodes
        if i_episode % 300 == 0:
            torch.save(ppo.policy.state_dict(), './ppo_model/ppo_{}.pth'.format(i_episode))
        '''

if __name__ == '__main__':
    main()
    




