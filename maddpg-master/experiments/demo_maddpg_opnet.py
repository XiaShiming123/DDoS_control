import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from experiments.env_RL import Env#
from gym import spaces
import copy
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=2, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    # from multiagent.environment import MultiAgentEnv
    # import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        #print("bbbbbbb------",i)
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        #print("cccccccc------", i)
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        EPISODE=8000


        # Create environment
        #env = make_env(arglist.scenario, arglist, arglist.benchmark)
        env = Env(n_nodes=108, max_flow=75000, name_='32_2000',attack_type=2)
        # Create agent trainers
        obs_shape_n = [env.observation_space.shape]
        num_adversaries = min(env.n, arglist.num_adversaries)
        #print("aaaaaaaaaa-------------------------------------------------------------------")
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)



        saver = tf.train.Saver()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        print('Starting iterations...')

        rs_train=[]
        for re_i in range(10):
            for k in range(8):
                s = env.reset(k, 1000 * k)
                epi_r = 0
                for e_i in range(1000):
                    aa = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

                    action_n = []
                    action_all = []
                    for i in range(env.agent_num):
                        obs = s[i]
                        obs = np.reshape(obs, (1, len(obs)))
                        temp_ = trainers[0].action(obs[0])
                        action_all.append(temp_)
                        temp_sum = 0.0
                        for j in range(len(aa)):
                            temp_sum += aa[j] * temp_[j] * 0.1
                        action_n.append(temp_sum)
                    new_obs_n, rew_n, done, normal_pass_rate = env.step(action_n)
                    if done == True:
                        s= env.reset(k, k * 1000 + i)
                        env.Us_pre_two_left = 0.0
                        env.Us_pre_left = 0.0
                        env.Us_now_left = 0.0
                        env.Us_now = env.Us
                        action_n = []
                        action_all = []
                        for i in range(env.agent_num):
                            obs = s[i]
                            obs = np.reshape(obs, (1, len(obs)))
                            temp_ = trainers[0].action(obs[0])
                            action_all.append(temp_)
                            temp_sum = 0.0
                            for j in range(len(aa)):
                                temp_sum += aa[j] * temp_[j] * 0.1
                            action_n.append(temp_sum)
                        s_, r, done, *_ = env.step(action_n)

                    rs_train.append(rew_n)
                    episode_step += 1
                    terminal = (episode_step >= arglist.max_episode_len)
                    if e_i % 100 == 0:
                        print("Train, {}, {}, reward:{}".format(k, e_i, rew_n))
                        #print(action_n, rew_n)
                        # collect experience
                    for i in range(env.agent_num):
                        trainers[0].experience(s[i], action_all[i], rew_n, new_obs_n[i], done, terminal)

                    s = new_obs_n


                    # increment global step counter
                    train_step += 1

                    # update all trainers, if not in display or benchmark mode
                    loss = None
                    for agent in trainers:
                        # print("loss:-------------------------")
                        agent.preupdate()
                    for agent in trainers:
                        loss = agent.update(trainers, train_step)
                        # print("loss:",loss)
        plt.figure()
        plt.plot(rs_train)
        plt.savefig("MADDPG-train-2_location_re2.png")
        rs_test = []
        crash_time=0
        for k in range(8, 10):
            s = env.reset(k, k * 1000)
            epi_r = 0
            for e_i in range(1000):
                aa = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                action_n = []
                action_all = []
                for i in range(env.agent_num):
                    obs = s[i]
                    obs = np.reshape(obs, (1,len(obs)))
                    temp_ = trainers[0].action(obs[0])
                    action_all.append(temp_)
                    temp_sum = 0.0
                    for j in range(len(aa)):
                        temp_sum += aa[j] * temp_[j] * 0.1
                    action_n.append(temp_sum)
                new_obs_n, rew_n, done, normal_pass_rate = env.step(action_n)
                if done == True:
                    crash_time+=1
                    s = env.reset(k, k * 1000 + i)
                    env.Us_pre_two_left = 0.0
                    env.Us_pre_left = 0.0
                    env.Us_now_left = 0.0
                    env.Us_now = env.Us
                    action_n = []
                    action_all = []
                    for i in range(env.agent_num):
                        obs = s[i]
                        obs = np.reshape(obs, (1, len(obs)))
                        temp_ = trainers[0].action(obs[0])
                        action_all.append(temp_)
                        temp_sum = 0.0
                        for j in range(len(aa)):
                            temp_sum += aa[j] * temp_[j] * 0.1
                        action_n.append(temp_sum)
                    new_obs_n, rew_n, done, normal_pass_rate = env.step(action_n)


                rs_test.append(rew_n)
                s = new_obs_n
                epi_r = 0.1 * rew_n + 0.9 * epi_r
                if e_i % 200 == 0:
                    print("test, {}, {}, reward:{}".format(k, e_i,rew_n))
    plt.figure()
    plt.plot(rs_test)
    plt.savefig("MADDPG-test-2_locationre2.png")
    print(np.mean(rs_test),crash_time)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
