#未训练时的马里奥
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True  #代表游戏结束了吗？true代表游戏结束
#总共5000帧
for step in range(5000):  
    if done:
        state = env.reset()  #启动游戏
    #每一步step返回状态，该步获得的奖励，游戏是否结束以及一些信息（是否到达终点、累计分数等）
    #state (240,256,3) 3通道的240*256
    state, reward, done, info = env.step(env.action_space.sample())  #动作随机采样
    #渲染
    env.render()

env.close()