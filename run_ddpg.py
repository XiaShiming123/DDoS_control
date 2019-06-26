import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
r_test_fair=np.load("r_test_fair.npy")
r_test_nofair=np.load("r_test_nofair.npy")
r_test_ddpg=np.load("ddpg_rs_test_MEM1600100.npy")
r_test_liner=np.load("liner_rs_test_0.npy")
r_test_maddpg=np.load("MADDPG_rs_test_.npy")
r_test_ddpg_LC=np.load("ddpg_com_rs_test_100.npy")

plt.figure(0)
plt.plot(r_test_nofair)
plt.title("SIRT")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
#plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("no_fair.png")

plt.figure(1)
plt.plot(r_test_fair)
plt.title("FRT")#:ave_reward="+str(np.round(np.mean(r_test_fair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("fair.png")


plt.figure(2)
plt.plot(r_test_ddpg)
plt.title("CRLRT")#:ave_reward="+str(np.round(np.mean(r_test_ddpg),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("ddpg.png")

plt.figure(3)
plt.plot(r_test_liner)
plt.title("Linear_Programming")#:ave_reward="+str(np.round(np.mean(r_test_liner),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("liner.png")

plt.figure(4)
plt.plot(r_test_maddpg)
plt.title("MART")#:ave_reward="+str(np.round(np.mean(r_test_maddpg),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("maddpg.png")

plt.figure(5)
plt.plot(r_test_ddpg_LC)
plt.title("CRLRT_LC")#:ave_reward="+str(np.round(np.mean(r_test_ddpg_LC),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.3,1.1])
plt.savefig("CRLRT_LC.png")
plt.show()

plt.plot(r_test_nofair, c="black")
plt.plot(r_test_fair, c="red")


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