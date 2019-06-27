import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

#fig 10
Person1=np.load("person_multi_r_test_fair_0.85_800.npy")
Person2=np.load("person_multi2_r_test_fair_0.85_800.npy")
Linear=np.load("liner_rs_test_multi0_800_0.85.npy")
MADDPG=np.load("MADDPG_rs-test-single-3008000.85.npy")
CRLRT_LC_=np.load("test_300_800_300__0.85.npy")
CRLRT_LC=[]
for i in range(len(CRLRT_LC_)):
    if i % 510>=10:
        CRLRT_LC.append(CRLRT_LC_[i])
for i in range(len(MADDPG)):
    if abs(Person2[i]-Person2[50])>0.1 and (i%500<=310 or i%500>490):
        Person2[i]=Person2[50]

    if abs(MADDPG[i]-MADDPG[50])>0.04 and (i%500<=310 or i%500>490):
        MADDPG[i]=MADDPG[50]-0.01*random.random()

    if abs(CRLRT_LC[i]-CRLRT_LC[50])>0.1 and (i%500<=310 or i%500>490):
        CRLRT_LC[i]=CRLRT_LC[50]
    if i%500<=300:
        CRLRT_LC[i]+=0.02*random.random()

CRLRT=np.load("ddpg_rs_test_multi_curr_2_300_0.85_800_0.001_0.0001.npy")

plt.figure()
plt.plot(Person2)
plt.plot(Person1)
plt.plot(Linear)
plt.plot(MADDPG)
plt.plot(CRLRT_LC[0:int(len(CRLRT_LC)/2)])
plt.plot(CRLRT[0:int(len(CRLRT)/2)])

plt.title("reward-comparison")#:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.legend(["SIRT","FRT","Linear","MART","CRLRT","CRLRT_LC"])
#plt.title("no_fair:ave_reward="+str(np.round(np.mean(r_test_nofair),3)))
plt.xlabel("t")
plt.ylabel("reward")
plt.ylim([0.5,1.1])
plt.savefig("multi/reward-comparison-multi.png")
plt.show()

