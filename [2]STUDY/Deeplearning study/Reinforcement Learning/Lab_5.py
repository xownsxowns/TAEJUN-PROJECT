import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Discount factor
dis = 0.99
num_episodes = 2000
learning_rate = 0.85

# create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using decay rate
        Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate*(reward + dis * np.max(Q[new_state,:]))

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: "+str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()