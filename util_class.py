#这段代码主要是为了在使用 Stable-Baselines3 (SB3) 训练强化学习模型（比如 PPO）时：
#自动保存表现最好的模型（SaveOnBestTrainingRewardCallback）；
#跳帧处理（SkipFrame）来加速训练；
#自定义奖励函数（RewardWrapper）来强化某些游戏行为（例如收金币、过关、失命）。

from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import gym
import os


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    # def _init_callback(self) -> None:
    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    # def _on_step(self) -> bool:
    #训练过程中表现最好的模型会被自动保存下来。
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model at {x[-1]} timesteps")
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(f'{self.save_path}/best_model{x[-1]}')

        return True

#跳帧
class SkipFrame(gym.Wrapper):
    """SkipFrame是可以实现跳帧操作。因为连续的帧变化不大，
    我们可以跳过n个中间帧而不会丢失太多信息。第n帧聚合每个跳过帧上累积的奖励。"""

    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
    #重写step函数，四帧返回一帧
    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

#重写reward函数
class RewardWrapper(gym.core.RewardWrapper):
    def __init__(self,env):
        super(RewardWrapper, self).__init__(env)
        self.coins = 0
        self.score = 0
        self.life=2

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # print('info:',info)
        info_dict=info
        coins = info_dict['coins']
        flag_get = info_dict['flag_get']
        score = info_dict['score']
        life = info_dict['life']

        # 如果coins大于self.coins, 奖励累加
        if coins > self.coins:
            reward += 200
            self.coins = coins
        # 如果flag_get为True, 奖励累加
        if flag_get:
            reward += 200

        if score > self.score:
            reward+=score-self.score
            self.score = score

        if life<self.life:
            reward-=500
            self.life=life

        if done:
            self.coins, self.score,self.life = 0, 0,3

        return observation, reward, done, info


