#coding=utf-8
#three kinds nodes edges wuhannode
#20180828
#39 agent 5  40 agent 4
import numpy as np
import copy
import random
import math
import pandas as pd
class Env_team():
    def __init__(self):
        self.one_episodes=270
        self.Us = 50000
        self.Ls = int(0.95 * self.Us)
        self.xishu_a = 0.5  # decrease
        self.xishu_b = 0.1  # increase
        self.pass_rate = 1.0
        self.numberofAgents=40

        self.opnet_data_Wuhan = pd.read_csv("DDOS_Proj-test1-DES-1__Wuhan.csv")
        temp_opnet_all_traffic_Wuhan = self.opnet_data_Wuhan.values
        self.opnet_all_traffic_Wuhan=copy.deepcopy(temp_opnet_all_traffic_Wuhan[50:-1])
        self.title_name_Wuhan = self.opnet_data_Wuhan.columns
        self.nodes_names_Wuhan = []
        for i in range(1, len(self.title_name_Wuhan)):
            temp_name = self.title_name_Wuhan[i]  # norm
            index0 = temp_name.index(".", 0, len(temp_name))
            index1 = temp_name.index("<", index0, len(temp_name))
            index2 = temp_name.index(">", index1, len(temp_name))
            index3 = temp_name.index("[", index2, len(temp_name))
            node0 = temp_name[index0 + 1:index1 - 1]
            node1 = temp_name[index2 + 2:index3 - 1]
            self.nodes_names_Wuhan.append([node0, node1])

        self.opnet_data_edges = pd.read_csv("DDOS_Proj-test1-DES-1__edges.csv")
        temp_opnet_all_traffic_edges = self.opnet_data_edges.values
        self.opnet_all_traffic_edges=copy.deepcopy(temp_opnet_all_traffic_edges[50:-1])
        self.title_name_edges = self.opnet_data_edges.columns
        self.nodes_names_edges = []
        for i in range(1, len(self.title_name_edges)):
            temp_name = self.title_name_edges[i]  # norm
            index0 = temp_name.index(":", 0, len(temp_name))
            index1 = temp_name.index("<", index0, len(temp_name))
            index2 = temp_name.index(">", index1, len(temp_name))
            index3 = temp_name.index("[", index2, len(temp_name))
            node0 = temp_name[index0 + 1:index1 - 1]
            node1 = temp_name[index2 + 2:index3 - 1]
            self.nodes_names_edges.append([node0, node1])

        self.opnet_data_nodes= pd.read_csv("DDOS_Proj-test1-DES-1__nodes.csv")
        temp_opnet_all_traffic_nodes = self.opnet_data_nodes.values
        self.opnet_all_traffic_nodes=copy.deepcopy(temp_opnet_all_traffic_nodes[50:-1])
        self.title_name_nodes=self.opnet_data_nodes.columns
        self.nodes_names_nodes=[]
        self.city_names=[]
        self.city_inside_names = []
        for i in range(1, len(self.title_name_nodes)):
            temp_name = self.title_name_nodes[i]  # norm
            index0 = temp_name.index(":", 0, len(temp_name))
            index0_1 = temp_name.index(".",index0, len(temp_name))
            index1 = temp_name.index("<", index0_1, len(temp_name))
            index2 = temp_name.index(">", index1, len(temp_name))
            index3 = temp_name.index("[", index2, len(temp_name))
            node0 = temp_name[index0 + 2:index0_1]
            node1 = temp_name[index0_1 + 1:index1-1]
            node2 = temp_name[index2 + 2:index3 - 1]
            self.nodes_names_nodes.append([node0,node1,node2])
            jishu = 0
            for j in range(len(self.city_names)):
                if self.city_names[j] == node0:
                    break
                jishu += 1
            if jishu == len(self.city_names):
                self.city_names.append(node0)
            jishu = 0
            for j in range(len(self.city_inside_names)):
                if self.city_inside_names[j] == node1:
                    break
                jishu += 1
            if jishu == len(self.city_inside_names):
                self.city_inside_names.append(node1)
            jishu = 0
            for j in range(len(self.city_inside_names)):
                if self.city_inside_names[j] == node2:
                    break
                jishu += 1
            if jishu == len(self.city_inside_names):
                self.city_inside_names.append(node2)
        a=0
    def reset(self):
        self.time = 0


    def get_time_date(self):
        self.state_all = []#---------------------------------------city nodes----------------------------------
        now_data_nodes_ = self.opnet_all_traffic_nodes[self.time%self.one_episodes]
        now_data_nodes = copy.deepcopy(now_data_nodes_[1:len(now_data_nodes_)])
        self.city_nodes=[[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
        for i in range(len(self.city_names)):
            for j in range(7):
                self.city_nodes[i][j]=now_data_nodes[2*(i*7+j)]+now_data_nodes[2*(i*7+j)+1]

        self.city_to_city=[]#---------------------------------------city to city----------------------------------
        now_data_edges_ = self.opnet_all_traffic_edges[self.time%self.one_episodes]
        now_data_edges = copy.deepcopy(now_data_edges_[1:len(now_data_edges_)])
        self.city_edges= [0]*int(len(self.nodes_names_edges)/2)
        for i in range(int(len(now_data_edges)/2)):
            self.city_edges[i]=now_data_edges[i*2]+now_data_edges[i*2+1]#---------------------------------------city to city----------------------------------

        # ---------------------------------------city wuhan----------------------------------
        now_data_Wuhan_ = self.opnet_all_traffic_Wuhan[self.time%self.one_episodes]
        now_data_Wuhan = copy.deepcopy(now_data_Wuhan_[1:len(now_data_Wuhan_)])
        self.city_Wuhan = [0] * int(len(self.nodes_names_Wuhan) / 2)
        for i in range(int(len(now_data_Wuhan) / 2)):
            self.city_Wuhan[i] = now_data_Wuhan[i * 2] + now_data_Wuhan[i * 2 + 1]
        # ---------------------------------------city wuhan----------------------------------

        # ---------------------------------------get state---------------------------------
        temp=[]

        self.agent_num=40
        self.normal_traffic = [0] *self.agent_num
        for i in range(self.agent_num):
            self.state_all.append(copy.deepcopy(temp))
        # for i in range(len(self.city_names)):
        #     self.state_all[i * 4 + 0].append(self.city_nodes[i][2])
        #     self.state_all[i * 4 + 0].append(self.city_nodes[i][0])
        #     self.state_all[i * 4 + 1].append(self.city_nodes[i][3])
        #     self.state_all[i * 4 + 1].append(self.city_nodes[i][0])
        #     self.state_all[i * 4 + 2].append(self.city_nodes[i][4])
        #     self.state_all[i * 4 + 2].append(self.city_nodes[i][1])
        #     self.state_all[i * 4 + 3].append(self.city_nodes[i][5])
        #     self.state_all[i * 4 + 3].append(self.city_nodes[i][1])
        self.state_all[0] = [self.city_nodes[0][3], self.city_nodes[0][1], self.city_edges[8], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[1] = [self.city_nodes[0][4], self.city_nodes[0][1], self.city_edges[8], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[2] = [self.city_nodes[0][5], self.city_nodes[0][2], self.city_edges[8], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[3] = [self.city_nodes[0][6], self.city_nodes[0][2], self.city_edges[8], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[4] = [self.city_nodes[1][3], self.city_nodes[1][1], self.city_edges[2], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[5] = [self.city_nodes[1][4], self.city_nodes[1][1], self.city_edges[2], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[6] = [self.city_nodes[1][5], self.city_nodes[1][2], self.city_edges[2], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[7] = [self.city_nodes[1][6], self.city_nodes[1][2], self.city_edges[2], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[8] = [self.city_nodes[2][3], self.city_nodes[2][1], self.city_edges[9], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[9] = [self.city_nodes[2][4], self.city_nodes[2][1], self.city_edges[9], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[10] = [self.city_nodes[2][5], self.city_nodes[2][2], self.city_edges[9], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[11] = [self.city_nodes[2][6], self.city_nodes[2][2], self.city_edges[9], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[12] = [self.city_nodes[3][3], self.city_nodes[3][1], self.city_edges[5], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[13] = [self.city_nodes[3][4], self.city_nodes[3][1], self.city_edges[5], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[14] = [self.city_nodes[3][5], self.city_nodes[3][2], self.city_edges[5], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[15] = [self.city_nodes[3][6], self.city_nodes[3][2], self.city_edges[5], self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[16] = [self.city_nodes[4][3], self.city_nodes[4][1], (self.city_edges[0]+self.city_edges[6]+self.city_edges[8]+self.city_edges[10])/4.0, self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[17] = [self.city_nodes[4][4], self.city_nodes[4][1], (self.city_edges[0] + self.city_edges[6]+self.city_edges[8] + self.city_edges[10])/4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[18] = [self.city_nodes[4][5], self.city_nodes[4][2], (self.city_edges[0] + self.city_edges[6]+self.city_edges[8] + self.city_edges[10])/4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[19] = [self.city_nodes[4][6], self.city_nodes[4][2], (self.city_edges[0] + self.city_edges[6]+self.city_edges[8] + self.city_edges[10])/4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[20] = [self.city_nodes[5][3], self.city_nodes[5][1], self.city_edges[6], self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[21] = [self.city_nodes[5][4], self.city_nodes[5][1], self.city_edges[6], self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[22] = [self.city_nodes[5][5], self.city_nodes[5][2], self.city_edges[6], self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[23] = [self.city_nodes[5][6], self.city_nodes[5][2], self.city_edges[6], self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[24] = [self.city_nodes[6][3], self.city_nodes[6][1],(self.city_edges[4] + self.city_edges[7]+self.city_edges[5] + self.city_edges[9]) / 4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[25]= [self.city_nodes[6][4], self.city_nodes[6][1],(self.city_edges[4] + self.city_edges[7]+self.city_edges[5] + self.city_edges[9]) / 4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[26]= [self.city_nodes[6][5], self.city_nodes[6][2],(self.city_edges[4] + self.city_edges[7]+self.city_edges[5] + self.city_edges[9]) / 4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[27]= [self.city_nodes[6][6], self.city_nodes[6][2],(self.city_edges[4] + self.city_edges[7]+self.city_edges[5] + self.city_edges[9]) / 4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[28] = [self.city_nodes[7][3], self.city_nodes[7][1],(self.city_edges[1] + self.city_edges[12]+self.city_edges[8] + self.city_edges[11]) / 4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[29] = [self.city_nodes[7][4], self.city_nodes[7][1],(self.city_edges[1] + self.city_edges[12]+self.city_edges[8] + self.city_edges[11]) / 4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[30]= [self.city_nodes[7][5], self.city_nodes[7][2],(self.city_edges[1] + self.city_edges[12]+self.city_edges[8] + self.city_edges[11]) / 4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[31]= [self.city_nodes[7][6], self.city_nodes[7][2],(self.city_edges[1] + self.city_edges[12]+self.city_edges[8] + self.city_edges[11]) / 4.0, self.city_Wuhan[2], self.city_Wuhan[3]]
        self.state_all[32] = [self.city_nodes[8][3], self.city_nodes[8][1], self.city_edges[11], self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[33] = [self.city_nodes[8][4], self.city_nodes[8][1], self.city_edges[11], self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[34] = [self.city_nodes[8][5], self.city_nodes[8][2], self.city_edges[11], self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[35] = [self.city_nodes[8][6], self.city_nodes[8][2], self.city_edges[11], self.city_Wuhan[2],self.city_Wuhan[3]]
        self.state_all[36] = [self.city_Wuhan[6], self.city_Wuhan[4], self.city_Wuhan[1],  self.city_Wuhan[3]]
        self.state_all[37] = [self.city_Wuhan[7], self.city_Wuhan[4], self.city_Wuhan[1],  self.city_Wuhan[3]]
        self.state_all[38] = [self.city_Wuhan[8], self.city_Wuhan[5], self.city_Wuhan[1],  self.city_Wuhan[3]]
        self.state_all[39] = [self.city_Wuhan[9], self.city_Wuhan[5], self.city_Wuhan[1],  self.city_Wuhan[3]]
        for i in range(len(self.state_all)):
            self.normal_traffic[i]=self.state_all[i][0]
        for i in range(len(self.state_all)-4):
            if i%4==3:
                self.normal_traffic[i]-=self.city_nodes[int(i/4)][6]
        self.normal_traffic[39]-=self.city_Wuhan[0]
        a=0

    def fram_step(self, action):
        reward=0
        pre_normal=0.0
        now_normal=0.0
        sun_now=0
        for i in range(len(action)):
            action[i]=round(action[i]*self.pass_rate)
            if action[i]>9:
                action[i]=9
            if action[i]<0:
                action[i]=0
        for i in range(len(action)):
            sun_now+=self.state_all[i][0]*(1-0.1*action[i])
            pre_normal+=self.normal_traffic[i]
            now_normal += self.normal_traffic[i]*(1-0.1*action[i])
        if sun_now>self.Us:
            reward=0
        else:
            if pre_normal==0:
                reward=0
            else:
                reward=now_normal/pre_normal

        self.time+= 1
        self.get_time_date()
        state_new = copy.deepcopy(self.state_all)
        if sun_now <= self.Ls:
            self.pass_rate = self.pass_rate * self.xishu_a
        if sun_now >= self.Us:
            self.pass_rate = self.pass_rate * (1 + self.xishu_b)
        return state_new,reward,action









if __name__ == '__main__':
    env=Env_team()
    env.reset()
    for i in range(20):
        env.time=50+i
        env.get_time_date()
        a=0