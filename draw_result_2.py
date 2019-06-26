# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np
#
# r_test_ddpg_LC=np.load("MADDPG_rs-test-single-1008000.85.npy")
# r_test_ddpg_LC2=np.load("MADDPG_rs_test-single-10020000.7.npy")
# show_length=1
# t=[]
# train_ddpg_show=[]
# train_ddpg_show2=[]
# for i in range(int(len(r_test_ddpg_LC)/show_length)):
#     t.append(int(i*show_length+show_length/2))
#     train_ddpg_show.append(np.mean(r_test_ddpg_LC[i*show_length:(i+1)*show_length]))
#     train_ddpg_show2.append(np.mean(r_test_ddpg_LC2[i * show_length:(i + 1) * show_length]))
# plt.figure(0)
# # plt.plot(t,train_ddpg_show2,'b')
# # #plt.plot(t,train_ddpg_show2,'k')
# # plt.title("train_CRLRT")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# # #plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# # plt.xlabel("t")
# # plt.ylabel("reward")
# # plt.ylim([0.3,1.1])
# # plt.savefig("train_CRLRT2.png")
#
#
#
# plt.plot(t, train_ddpg_show, 'b')
# plt.plot(t,train_ddpg_show2,'k')
# plt.title("MART")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.legend(["800_0.85","2000_0.7"])
# #plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("MART.png")
# plt.show()
#
#


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# r_test_ddpg_LC=np.load("ddpg_single_train_900_0.85_800_0.001_0.0001.npy")
# r_test_ddpg_LC2=np.load("ddpg_single_train_300_0.85_800_0.001_0.0001.npy")
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
# plt.plot(t,train_ddpg_show2, 'b')
# plt.plot(t,train_ddpg_show,'k')
# plt.title("CRLRT_800_0.85")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.legend(["curriculum learning","directly learning"])
# #plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("curriculum-directly_train_single.png")
# plt.show()

r_test_ddpg_LC=np.load("ddpg_rs_test_single_dire_900_0.85_800_0.001_0.0001.npy")
r_test_ddpg_LC2=np.load("ddpg_rs_test_single300_0.85_800_0.001_0.0001.npy")
show_length=5
t=[]
train_ddpg_show=[]
train_ddpg_show2=[]
for i in range(int(len(r_test_ddpg_LC)/2/show_length)):
    t.append(int(i*show_length+show_length/2))
    train_ddpg_show.append(np.mean(r_test_ddpg_LC[i*show_length:(i+1)*show_length]))
    train_ddpg_show2.append(np.mean(r_test_ddpg_LC2[i * show_length:(i + 1) * show_length]))
plt.figure(0)

plt.plot(r_test_ddpg_LC2[0:int(len(r_test_ddpg_LC)/2)], 'b')
plt.plot(t,train_ddpg_show,'k')
plt.title("CRLRT_800_0.85")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.legend(["curriculum learning","directly learning"])
#plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("curriculum-directly.png")
plt.show()

# plt.plot(t,train_ddpg_show2,'b')
# #plt.plot(t,train_ddpg_show2,'k')
# plt.title("train_CRLRT")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# #plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("train_CRLRT2.png")



# plt.plot(t, train_ddpg_show2, 'b')
# plt.plot(t,train_ddpg_show,'k')
# plt.title("reward_compare")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.legend(["our_reward_function","prior_reward_function"])
# #plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("reward_compare_single.png")
# plt.show()


