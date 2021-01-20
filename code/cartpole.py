import gym
import pyvirtualdisplay

env = gym.make("CartPole-v1")
obs = env.reset()
print(obs)

display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
env.render()

print(env.action_space)
action = 1
obs, reward, done, info = env.step(action)
print(reward)