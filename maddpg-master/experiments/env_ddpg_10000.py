import numpy as np
from queue import Queue
import copy
from gym import spaces


class Env:
    def __init__(self, n_routers=108, n_attackers=54, normal_uniform=False, random_seed=1):

        np.random.seed(random_seed)
        self.n = 1

        self.n_routers = n_routers
        self.n_attackers = n_attackers
        action_dim = 10
        self.action_space = [spaces.Discrete(action_dim)]
        # self.lowlist1 = [0, 1000, 5000, 10000, 70000]
        # self.highlist1 = [3000, 9000, 27000, 80000, 120000]
        self.lowlist1 = [0,0, 0.2, 3, 10, 40]
        self.highlist1 = [1, 1, 4,12, 36, 90]
        high = np.array(self.highlist1)
        low = np.array(self.lowlist1)
        self.observation_space = spaces.Box(low=low, high=high)

        normal_mean = 800
        attack_mean = 800
        self.upper_bound =(normal_mean * n_routers + attack_mean * n_attackers) * 0.85# 80000
        self.router = Router(n_routers, mean=normal_mean, uniform=normal_uniform)
        self.attacker = Attacker(n_attackers=n_attackers, n_routers=n_routers, mean=attack_mean)
        self.server = Server(self.upper_bound)

        self.normalizer = (normal_mean + attack_mean) * 1.1
        self.normal_jishu=0

    def reset(self):
        self.router.reset()
        self.attacker.reset()
        self.server.reset()
        self.server.is_being_attack=0

        normal = self.router.send()
        attack = self.attacker.attack()
        state = normal + attack
        server_x = self.upper_bound


        self.state_all = copy.deepcopy(state/self.normalizer)
        self.normal_all = copy.deepcopy(normal/self.normalizer)

        for i in range(self.n_routers):  # scale to 0~1
            self.state_all[i] = np.round(self.state_all[i], 4)
            self.normal_all[i] = np.round(self.normal_all[i], 4)
        net_struct = [4, 3, 3, 3]
        get_state_all = []
        temp = []
        temp_1 = []
        for i in range(int(self.n_routers / net_struct[3])):
            temp_1.append(sum(self.state_all[i * net_struct[3]:(i + 1) * net_struct[3]]))
        temp_2 = []
        for i in range(int(len(temp_1) / net_struct[2])):
            temp_2.append(sum(temp_1[i * net_struct[2]:(i + 1) * net_struct[2]]))
        temp_3 = []
        for i in range(int(len(temp_2) / net_struct[1])):
            temp_3.append(sum(temp_2[i * net_struct[1]:(i + 1) * net_struct[1]]))
        for i in range(self.n_routers):
            get_state_all.append([i / 108.0, self.state_all[i], temp_1[int(i / net_struct[3])],
                                  temp_2[int(int(i / net_struct[3]) / net_struct[2])],
                                  temp_3[int(int(int(i / net_struct[3]) / net_struct[2]) / net_struct[1])],
                                  sum(self.state_all)])

        return copy.deepcopy(get_state_all)

    def step(self, action):

        self.attacker.env_time += 1
        if self.server.is_being_attack == 0:
            action = np.zeros(self.n_routers)

        normal = self.router.send()
        attack = self.attacker.attack()
        data_4_server = []
        now_sum = 0.0
        for i in range(self.n_routers):
            now_sum += normal[i] + attack[i]
            data_4_server.append({"normal": normal[i], "attack": attack[i], "action": action[i]})

        if now_sum > self.upper_bound:
            self.server.is_being_attack = 1
        if self.server.is_being_attack==1:
            if now_sum<self.upper_bound:
                self.normal_jishu+=1
                if self.normal_jishu>10:
                    self.server.is_being_attack = 0
                    self.normal_jishu=0

        server_x, reward,done = self.server.step(data_4_server)
        state = normal + attack
        state_next = np.zeros(self.n_routers)
        state_next[:self.n_routers] = state / self.normalizer
        self.state_all=copy.deepcopy(state/self.normalizer)
        self.normal_all = copy.deepcopy(normal/ self.normalizer)

        for i in range(self.n_routers):  # scale to 0~1
            self.state_all[i] = np.round(self.state_all[i], 4)
            self.normal_all[i] = np.round(self.normal_all[i], 4)
        net_struct = [4, 3, 3, 3]
        get_state_all = []
        temp = []
        temp_1 = []
        for i in range(int(self.n_routers / net_struct[3])):
            temp_1.append(sum(self.state_all[i * net_struct[3]:(i + 1) * net_struct[3]]))
        temp_2 = []
        for i in range(int(len(temp_1) / net_struct[2])):
            temp_2.append(sum(temp_1[i * net_struct[2]:(i + 1) * net_struct[2]]))
        temp_3 = []
        for i in range(int(len(temp_2) / net_struct[1])):
            temp_3.append(sum(temp_2[i * net_struct[1]:(i + 1) * net_struct[1]]))
        for i in range(self.n_routers):
            get_state_all.append([i / 108.0, self.state_all[i], temp_1[int(i / net_struct[3])],
                                  temp_2[int(int(i / net_struct[3]) / net_struct[2])],
                                  temp_3[int(int(int(i / net_struct[3]) / net_struct[2]) / net_struct[1])],
                                  sum(self.state_all)])
        # self.state_all.append(self.Us_now/self.guiyihua)
        # self.state_all.append(self.Us / self.guiyihua)

        return copy.deepcopy(get_state_all), reward,action,done


class Router:
    def __init__(self, n_routers, uniform, mean=625, var=200):
        self.n_routers = n_routers
        self.mean = mean
        self.var = var

        self.normalizer = mean
        self.uniform = uniform

        if not self.uniform:
            self.small_location = [4, 5, 6, 7, 32, 33, 34, 35, 52, 53, 54, 55, 72, 73, 74, 75, 98, 99, 100, 101]
            self.bigger_location = [0, 1, 2, 3, 12, 13, 14, 15, 40, 41, 42, 43, 60, 61, 62, 63, 80, 81, 82, 83]

    def reset(self):
        pass

    def send(self):
        normal = np.maximum(np.random.normal(loc=self.mean, scale=self.var, size=self.n_routers), 0)
        if not self.uniform:
            normal[self.small_location] *= 0.7
            normal[self.bigger_location] *= 1.3
        return normal


class Attacker:
    def __init__(self, n_attackers, n_routers, mean=2000, var=100):
        self.n_attackers = n_attackers
        self.n_routers = n_routers
        self.mean = mean
        self.var = var

        self.location = np.random.randint(0, self.n_routers, size=n_attackers)
        self.env_time = 0

    def reset(self):
        self.location = np.random.randint(0, self.n_routers, size=self.n_attackers)
        self.env_time = 0

    def attack(self):
        x = np.zeros(self.n_routers)
        if self.env_time % 10000 >= 8000 and self.env_time % 10000 < 8300:
            x[self.location] = np.random.normal(self.mean, self.var, size=self.n_attackers)
            x = np.maximum(x, 0)
        # x[self.location] = np.random.normal(self.mean, self.var, size=self.n_attackers)
        # x = np.maximum(x, 0)
        return x


class Server:
    def __init__(self, upper_bound=80000):
        self.is_being_attack = 0
        self.upper_bound = upper_bound
        self.upper_bound_now=upper_bound

        self.pool = Queue()
        self.r1 = 0


    def reset(self):
        self.pool = Queue()
        self.r1 = 0

    def step(self, state):
        for data in state:
            self.pool.put(data)
        normal_process = 0
        normal_total = 0
        #x = self.upper_bound_now
        # t-2
        now_total=0.0
        while not self.pool.empty():
            data = self.pool.get()
            now_total+= (data["normal"] + data["attack"]) * (1 - data["action"])
            normal_process += data["normal"] * (1 - data["action"])
            normal_total += data["normal"]

        if now_total>self.upper_bound_now:
            r2 = normal_process / normal_total*self.upper_bound_now/now_total
        else:
            r2 = normal_process / normal_total

        self.upper_bound_now = min(2*self.upper_bound_now-now_total,self.upper_bound)
        #print(self.upper_bound_now)


        # 单奖励

        reward = r2 - self.r1

        self.r1 = r2



        done=False
        if self.upper_bound_now<0:
            done=True
        return self.upper_bound_now, reward,done


if __name__ == "__main__":
    env = Env(n_routers=10, n_attackers=2)

    s = env.reset()
    print(s)
    for i in range(3):
        a = np.random.uniform(0, 1, 10)

        s_, r = env.step(a)
        print(r)

