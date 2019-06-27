import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
person_single_r_test_7=np.load("person_single_r_test_fair_0.7_2000.npy")
person_single_r_test_85=np.load("person_single_r_test_fair_0.85_800.npy")

plt.figure(0)

plt.plot(person_single_r_test_85,c='blue')
plt.plot(person_single_r_test_7,c='black')

plt.title("SIRT")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.legend(["800_0.85","2000_0.7",  "q_size"])
#plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("single/person_single1.png")


person_single2_r_test_7=np.load("person_single2_r_test_fair_0.7_2000.npy")
person_single2_r_test_85=np.load("person_single2_r_test_fair_0.85_800.npy")

plt.figure(1)

plt.plot(person_single2_r_test_85,c='blue')
plt.plot(person_single2_r_test_7,c='black')

plt.title("SIRT")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.legend(["800_0.85","2000_0.7", "q_size"])
#plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("single/person_single2.png")


ddpg_single_rs1=np.load("ddpg_rs_test_single999_0.7_2000_0.001_0.0001.npy")
ddpg_single_rs2=np.load("ddpg_rs_test_single300_0.85_800_0.001_0.0001.npy")

plt.figure(2)

plt.plot(ddpg_single_rs2[0:5000],c='blue')
plt.plot(ddpg_single_rs1[0:5000],c='black')

plt.title("CRLRT")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.legend(["800_0.85","2000_0.7", "q_size"])
#plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("single/ddpg___.png")


ddpg_single_rs1=np.load("liner_rs_test_0_2000_0.7.npy")
ddpg_single_rs2=np.load("liner_rs_test_0_800_0.85.npy")
plt.figure(3)
plt.plot(ddpg_single_rs1[0:5000],c='blue')
plt.plot(ddpg_single_rs2[0:5000],c='black')
plt.title("Linear")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.legend(["2000_0.7","800_0.85","2000_0.7", "q_size"])
#plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("single/linear.png")

# plt.figure(1)
# plt.plot(r_test_fair)
# plt.title("FRT")#:ave_reward="+str(np.round(np.mean(r_test_fair),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("fair.png")
#
#
# plt.figure(2)
# plt.plot(r_test_ddpg)
# plt.title("CRLRT")#:ave_reward="+str(np.round(np.mean(r_test_ddpg),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("ddpg.png")
#
# plt.figure(3)
# plt.plot(r_test_liner)
# plt.title("Linear_Programming")#:ave_reward="+str(np.round(np.mean(r_test_liner),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("liner.png")
#
# plt.figure(4)
# plt.plot(r_test_maddpg)
# plt.title("MART")#:ave_reward="+str(np.round(np.mean(r_test_maddpg),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("maddpg.png")
#
# plt.figure(5)
# plt.plot(r_test_ddpg_LC)
# plt.title("CRLRT_LC")#:ave_reward="+str(np.round(np.mean(r_test_ddpg_LC),3)))
# plt.xlabel("t")
# plt.ylabel("reward")
# plt.ylim([0.3,1.1])
# plt.savefig("CRLRT_LC.png")
# plt.show()
#
# plt.plot(r_test_nofair, c="black")
# plt.plot(r_test_fair, c="red")
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