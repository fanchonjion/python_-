#强化学习了2000次后，对100名乘客进行测试,打印运送的成功率
import gym
import numpy as np
env = gym.make("Taxi-v2")
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
alpha = 0.618
r = []
for episode in range(1,2001):
    done = False
    G, reward = 0,0
    state = env.reset()
    while done != True:
        action = np.argmax(Q[state]) #1
        state2, reward, done, info = env.step(action) #2
        Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
        G += reward
        state = state2    
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))
k = 0
for i in range(100):
    state = env.reset()
    reward = 0
    done = None
    while done !=True:
        action = np.argmax(Q[state])
        state,reward,done,info = env.step(action)
        #env.render()
    if reward==20:
        k = k+1
print('运送乘客的成功率:{0}%'.format(k))
