# 依赖安装
```bash
python 3.8
pip install -r requirements.txt
setuptools==65.5.0
wheel==0.38.4

gym==0.21.0
gym-super-mario-bros==7.3.0

tensorboard
six
protobuf==3.20.3
icecream
imageio
opencv-python
nes_py==8.2.1
numpy==1.24.4

shimmy>=2.0
stable_baselines3==1.6.0
```


# 项目执行
一个入门级强化学习项目
```bash
python .\test\test_mario.py  未训练的Mario展示
python .\test_model.py  测试模型在Mario的效果
python .\train.py       模型训练找到策略
```
