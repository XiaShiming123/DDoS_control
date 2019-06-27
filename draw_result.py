import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#fig 9
# r_test_ddpg_LC=np.load("ddpg_single_train_300_0.85_800_0.001_0.0001.npy")
# r_test_ddpg_LC2=np.load("ddpg_single_train_curr_pri_reward300_0.85_800_0.001_0.0001.npy")
# show_length=1000
# t=[]
# train_ddpg_show=[]
# train_ddpg_show2=[]
# for i in range(int(len(r_test_ddpg_LC)/2/show_length)):
#     t.append(int(i*show_length+show_length/2))
#     train_ddpg_show.append(np.mean(r_test_ddpg_LC[i*show_length:(i+1)*show_length]))
#     train_ddpg_show2.append(np.mean(r_test_ddpg_LC2[i * show_length:(i + 1) * show_length]))
# plt.figure(0)
#
# plt.plot(t,train_ddpg_show,'b')
# plt.plot(t,train_ddpg_show2, 'k')
#
# plt.title("CRLRT_800_0.85")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.legend(["new_reward","prior_reward"])
# #plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("reward_single.png")
# plt.show()



# r_test_ddpg_LC=np.load("ddpg_rs_test_single300_0.85_800_0.001_0.0001.npy")
# r_test_ddpg_LC2=np.load("ddpg_single_train_300_0.85_800_0.001_0.0001.npy")
# show_length=1
# t=[]
# train_ddpg_show=[]
# train_ddpg_show2=[]
# for i in range(int(len(r_test_ddpg_LC)/2/show_length)):
#     if i%500<300:
#         t.append(int(i*show_length+show_length/2))
#         train_ddpg_show.append(np.mean(r_test_ddpg_LC[i*show_length:(i+1)*show_length]))
#         train_ddpg_show2.append(np.mean(r_test_ddpg_LC2[i * show_length:(i + 1) * show_length]))
# plt.figure(0)
# plt.plot(t, train_ddpg_show, 'b')
# plt.title("MART_800_0.85")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.savefig("MART_train.png")
# #plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.figure(1)
# plt.plot(t, train_ddpg_show2, 'b')
# plt.title("CRLRT_800_0.85")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.savefig("CRLRT_train.png")
# plt.show()

r_test_ddpg_LC=np.load("ddpg_rs_test_single_pri_reward900_0.85_800_0.001_0.0001.npy")
r_test_ddpg_LC2=np.load("ddpg_rs_test_single_curr_pri_reward300_0.85_800_0.001_0.0001.npy")
show_length=1
t=[]
train_ddpg_show=[]
train_ddpg_show2=[]
for i in range(int(len(r_test_ddpg_LC)/show_length)):
    t.append(int(i*show_length+show_length/2))
    train_ddpg_show.append(np.mean(r_test_ddpg_LC[i*show_length:(i+1)*show_length]))
    train_ddpg_show2.append(np.mean(r_test_ddpg_LC2[i * show_length:(i + 1) * show_length]))
plt.figure(0)

plt.plot(t, train_ddpg_show, 'b')
plt.plot(t,train_ddpg_show2,'k')
plt.title("CRLRT_800_0.85")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.legend(["directly","Curriculum", "q_size"])
#plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("prireward_test.png")
plt.show()


# #fig 6(a) (b)
# r_test_ddpg_LC=np.load("MADDPG_rs_train-single-1008000.85.npy")
# r_test_ddpg_LC2=np.load("ddpg_single_train_300_0.85_800_0.001_0.0001.npy")
# show_length=1
# t=[]
# train_ddpg_show=[]
# train_ddpg_show2=[]
# for i in range(int(len(r_test_ddpg_LC)/show_length)):
#     if i%500<300:
#         t.append(int(i*show_length+show_length/2))
#         train_ddpg_show.append(np.mean(r_test_ddpg_LC[i*show_length:(i+1)*show_length]))
#         train_ddpg_show2.append(np.mean(r_test_ddpg_LC2[i * show_length:(i + 1) * show_length]))
# plt.figure(0)
# plt.plot(t, train_ddpg_show, 'b')
# plt.title("MART_800_0.85")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.savefig("MART_train.png")
# #plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.figure(1)
# plt.plot(t, train_ddpg_show2, 'b')
# plt.title("CRLRT_800_0.85")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.savefig("CRLRT_train.png")
# plt.show()




# plt.plot(np.array(qsize)/N_AGENTS/4, c="blue")
# plt.legend(["Reward", "Rate", "q_size"])
# plt.savefig("强化学习/train-" + str(REWARD_ALPHA) + str(regularization) + "_0.png")
#
# plt.figure()
# plt.plot(rate, c="black")
# plt.plot(np.array(qsize)/N_AGENTS/4, c="blue")
# plt.legend(["Rate", "q_size"])
# plt.title(np.mean(rate))
# plt.savefig("强化学习/test-" + str(REWARD_ALPHA) + str(regularization) + "_0.png")
#
#
#
# plt.plot(rs, c="red")
# plt.plot(rate, c="black")
# plt.plot(np.array(qsize)/N_AGENTS/4, c="blue")
# plt.legend(["Reward", "Rate", "q_size"])