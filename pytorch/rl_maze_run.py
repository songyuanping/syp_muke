"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
from rl_maze_brain import QLearningTable
from rl_maze_env import Maze


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, down = env.step(action)
            # RL learn from this transsition
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if down:
                break
    # end if game
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
