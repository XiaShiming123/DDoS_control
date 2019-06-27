import numpy as np
import copy
import random
import math
import pandas as pd
import numpy as np
import numpy as np
import copy
import random
import math
import pandas as pd
from gym import spaces

class Env:
    def __init__(self, n_nodes, max_flow,name_,attack_type):
        """
        Environment
        :param n_nodes: number of node; int
        :param max_flow: max flow for server; int
        
        
        """
        self.n = 1
        self.one_episodes = 1000
        self.Us = 50000
        self.Ls = int(0.95 * self.Us)
        # self.xishu_a = 0.5  # decrease
        # self.xishu_b = 0.1  # increase
        # self.pass_rate = 1.0
        #self.numberofAgents = 108
        self.agent_num=n_nodes
        action_dim = 10
        self.action_space = [spaces.Discrete(action_dim)]
        # self.lowlist1 = [0, 1000, 5000, 10000, 70000]
        # self.highlist1 = [3000, 9000, 27000, 80000, 120000]
        self.lowlist1 = [0,0, 0.2, 1, 4, 10]
        self.highlist1 = [1,1, 2.5, 7, 14, 35]
        high = np.array(self.highlist1)
        low = np.array(self.lowlist1)
        self.observation_space = spaces.Box(low=low, high=high)

        self.attack_type=attack_type
        self._n_nodes = n_nodes#the upload for server
        self.Us=max_flow
        self.Us_now = max_flow
        self.Us_pre = 0.0
        self.Us_pre_two = 0.0
        self.Us_now_left = 0.0
        self.Us_pre_left = 0.0
        self.Us_pre_two_left = 0.0
        self._var = max_flow * 0.1
        self._mean = max_flow * 0.5
        self._state = np.random.normal(loc=self._mean, scale=self._var, size=n_nodes)
        self.action_range = [0, 1]
        self.guiyihua=3000.0
        self.Us_jishu = 0  # if the left Us left for three times


        #load data from csv
        self.attack_agents_locations = np.load("attack_agents_locations_"+name_+".npy")#load attack agent location
        self.opnet_all_traffic_nodes = np.load("normal_data_.npy")  #

        self.date_len=10000# the length of all normal data from csv file

        self.attack_agent_num=len(self.attack_agents_locations[0])#attack_occured_in_agent
        self.attack_agent=[0]*self.attack_agent_num
        self.attack_traffic=int(name_[3:7])

    def reset(self, repeat_i,begin):
        """
        Reset environment
        :param repeat_i: i th repeat data
        :return: initial flow value; array, (n_nodes, )
        """

        self.time = begin
        for i in range(self.attack_agent_num):
            self.attack_agent[i] = self.attack_agents_locations[repeat_i%len(self.attack_agents_locations)][i]
        _state= self._get_time_date()
        return _state

    def step(self, action):
        """
        Accept joint action and Return next state and reward
        :param action: joint action; array, (n_nodes, )
        :return: next state; array, (n_nodes, )
                 reward; float
                 done; boolean
        """
        # receive action, return reward and next state
        self._state, reward,done,normal_pass_rate = self._get_next(action)
        return self._state, reward,done,normal_pass_rate


    def _get_next(self, action):
        """
        Get next state and reward
        :param action: joint action; array, (n_nodes, )
        :return: next_state; array, (n_node, )
                 reward; float
        """
        reward = 0.0
        sun_now = 0.0
        pre_total = 0
        now_total = 0
        self.now_time_action = copy.deepcopy(action)
        self.Us_pre_two = copy.deepcopy(self.Us_pre)
        self.Us_pre = copy.deepcopy(self.Us_now)


        for i in range(self._n_nodes):
            sun_now += self.now_time_total_traffic[i] * (1 - self.now_time_action[i])
            now_total += self.now_time_normal_traffic[i] * (1 - self.now_time_action[i])
            pre_total += self.now_time_normal_traffic[i]

        normal_pass_rate=0
        if sun_now >= self.Us_now:#--------more than server--------------
            if pre_total == 0:#--reward--
                reward = 0
            else:
                reward = now_total / pre_total*(self.Us_now/sun_now)
            self.Us_now_left = sun_now - self.Us_now
            #reward = -(sun_now - self.Us_now) / self.Us#reward less than 0
            reward = np.round(reward, 4)
            normal_pass_rate=0.0
            #self.Us_now = 2 * self.Us_now - sun_now
            self.Us_jishu += 1#--------more than server------------------
        else:#---------------------less than server--------------------------------------------
            self.Us_jishu=0
            self.Us_now_left = 0.0
            Us_pre_two_add = self.Us_now - sun_now
            if Us_pre_two_add<self.Us_pre_two_left:
                self.Us_pre_two_left-=Us_pre_two_add
            else:
                self.Us_pre_two_left=0.0
                Us_pre_add=Us_pre_two_add-self.Us_pre_two_left
                if Us_pre_add <self.Us_pre_left:
                    self.Us_pre_left-=Us_pre_add
                else:
                    self.Us_pre_left=0.0

            if pre_total == 0:#--reward--
                reward = 0
            else:
                reward = now_total / pre_total
            reward = np.round(reward, 4)#--reward--
            normal_pass_rate=reward#---------------------less than server----------------------

        self.Us_now = self.Us - self.Us_now_left - self.Us_pre_left - self.Us_pre_two_left
        self.time += 1
        next_state = self._get_time_date()
        done = False
        if self.Us_now < self.Us / 3.0:
            done = True
        return next_state, reward, done,normal_pass_rate

    def _get_time_date(self):
        """
        Get now state
        :param pre_cut_traffic: left traffic from prior time; list [n_node]
        :return: now_state; array, (n_node, )
        """

        self.now_time_normal_traffic=[]#get normal traffic from csv
        now_data_nodes_ = self.opnet_all_traffic_nodes[self.time % 10000]
        for i_ in range(len(now_data_nodes_)):
            if i_ % 5 == 0:
                self.now_time_normal_traffic.append(now_data_nodes_[i_] * 1.5)
            else:
                if i_ % 3 == 0:
                    self.now_time_normal_traffic.append(now_data_nodes_[i_] * 1.0)
                else:
                    self.now_time_normal_traffic.append(now_data_nodes_[i_]*0.5)
        self.now_time_total_traffic = copy.deepcopy(self.now_time_normal_traffic)


        if self.attack_type == 2:
            T = 20
            if self.time > T and self.time % T < T / 2:
                for i in range(len(self.attack_agent)):  # add attack traffic
                    self.now_time_total_traffic[self.attack_agent[i]] += self.attack_traffic
            if self.time > T and self.time % T >= T / 2:
                for i in range(len(self.attack_agent)):  # add attack traffic
                    self.now_time_total_traffic[self.attack_agent[i]] += self.attack_traffic/2

        self.state_all = copy.deepcopy(self.now_time_total_traffic)
        self.normal_all = copy.deepcopy(self.now_time_normal_traffic)
        #print(np.std(self.state_all))

        for i in range(self._n_nodes):#scale to 0~1
            self.state_all[i]/=self.guiyihua
            self.normal_all[i]/=self.guiyihua
            self.state_all[i] = np.round(self.state_all[i], 4)
            self.normal_all[i]=np.round(self.normal_all[i],4)
        net_struct=[4,3,3,3]
        get_state_all=[]
        temp = []
        temp_1=[]
        for i in range(int(self.agent_num/net_struct[3])):
            temp_1.append(sum(self.state_all[i*net_struct[3]:(i+1)*net_struct[3]]))
        temp_2 = []
        for i in range(int(len(temp_1)/net_struct[2])):
            temp_2.append(sum(temp_1[i*net_struct[2]:(i+1)*net_struct[2]]))
        temp_3 = []
        for i in range(int(len(temp_2) / net_struct[1])):
            temp_3.append(sum(temp_2[i * net_struct[1]:(i + 1) * net_struct[1]]))
        for i in range(self.agent_num):
            get_state_all.append([i/108.0,self.state_all[i],temp_1[int(i/net_struct[3])],temp_2[int(int(i/net_struct[3])/net_struct[2])],
                                  temp_3[int(int(int(i/net_struct[3])/net_struct[2])/net_struct[1])],sum(self.state_all)])
        # self.state_all.append(self.Us_now/self.guiyihua)
        # self.state_all.append(self.Us / self.guiyihua)

        return np.array(copy.deepcopy(get_state_all))
