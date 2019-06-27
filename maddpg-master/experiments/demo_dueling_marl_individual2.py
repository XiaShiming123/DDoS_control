#20180828
#dueling DQN  eaach agent with one network
#270

import os
from RL_brain import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from env_opnet_0820 import Env_team#
import copy


MEMORY_SIZE = 300
Learning_SIZE=10
ACTION_SPACE = 10
EPISODE = 2700

sess = tf.Session()

def return_DQN(name_):
    with tf.variable_scope(name_):
        dueling_DQN = DuelingDQN(
            n_actions=ACTION_SPACE, n_features=5, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)
    return dueling_DQN
dueling_DQN_40={}
for i in range(40):
    temp_dueling_DQN=return_DQN('dueling'+str(i))
    dueling_DQN_40[str(i)]=temp_dueling_DQN

sess.run(tf.global_variables_initializer())


def train(RL,saver,saved_net,RL_greedy):

    length_show=10

    t = []

    e_greedy_increment_ = (1-RL_greedy)/(EPISODE * 0.8)
    actions_all = []
    rewards_all = []
    buffer_count = 0
    episode_rewards = [0.0]

    for episode in range(EPISODE):
        print("episode:",episode)
        # env.get_time_date()
        state_all = env.state_all
        action = []
        for i in range(env.numberofAgents):
            action.append(RL[str(i)].choose_action([state_all[i]], RL_greedy))  # e-greedy action for train
        new_obs, rew, action = env.fram_step(action)
        actions_all.append(action)
        rewards_all.append(rew)

        for i in range(env.numberofAgents):
            RL[str(i)].store_transition(state_all[i], action[i], rew, new_obs[i])
            buffer_count += 1
        episode_rewards[-1] += rew


        # print(episode, env.time)

        env.time += 1
        env.get_time_date()


        RL_greedy=RL_greedy+e_greedy_increment_
        #print(1-RL_greedy)
        env.reset()

        if buffer_count % 100 == 0:
            for i in range(env.numberofAgents):
                saver[i].save(sess, saved_net[i] + 'network' + '-dqn', global_step=buffer_count)
            if buffer_count > Learning_SIZE:  # learning
                for i in range(env.numberofAgents):
                    RL[str(i)].learn()

    t = range(EPISODE)
    show_length = 10
    t_ = []

    pass_rate_ = []

    for i in range(0, int(EPISODE / show_length) - 1):
        # print(i)
        t_.append(t[i * show_length])
        pass_rate_.append(sum(rewards_all[i * show_length:(i + 1) * show_length]) / show_length)
    plt.figure(2)
    plt.title("MARL_dueling_DQN", fontsize='large')
    plt.xlabel('episode', fontsize='large')
    plt.ylabel('reward', fontsize='large')
    plt.plot(t_, pass_rate_)
    plt.show()

if __name__ == '__main__':
    tic = time.clock()
    print('begin_time:', tic)
    env = Env_team()
    env.reset()
    env.time = 0
    env.get_time_date()
    saver_40=[]
    path_name_40=[]
    for i in range(40):
        path_name = "saved_networks_opnet_"+str(i)
        folder = os.path.exists(path_name)
        if not folder:
            os.makedirs(path_name)
        saver = tf.train.Saver()
        saver_40.append(saver)

        checkpoint = tf.train.get_checkpoint_state(path_name)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        path_name_40.append(path_name+'/')
    train(dueling_DQN_40,saver_40,path_name_40,0.1 )
        # np.save('q_natural', q_natural)
    toc = time.clock()
    print('end_time:', toc)
    print('total_time:', (toc - tic) / 3600.0)


