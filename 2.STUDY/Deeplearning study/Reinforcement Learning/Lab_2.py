import gym.spaces
from gym.envs.registration import register
import msvcrt

class _Getch:
    def __call__(self):
        keyy = msvcrt.getch()
        return msvcrt.getch()

inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    b'H': UP,
    b'P': DOWN,
    b'M': RIGHT,
    b'K': LEFT
}

register(
    id='FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery':False}
)

env = gym.make('FrozenLake-v3')
env.render()

while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break