# DCGAN

![image](https://user-images.githubusercontent.com/45096827/149706048-1523f22d-5fac-4b52-a466-88125b5b1bcb.png)


# 1.Installation

```
nvidia-docker run --name GAN -it -v /your/project/path/:/project/path/ --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

apt update

apt install -y libgl1-mesa-glx

pip uninstall opencv-python
pip install opencv-python
```
***

# 2.Training with Testing

you can change parameters using [config.py](https://github.com/josh3255/GAN/blob/master/config.py)

```
cd /project/path
python train.py
```

***

# 3.Features

![image](https://user-images.githubusercontent.com/45096827/149706538-41f66ddd-1a9a-434c-83e5-16bb9f00f36c.png)


***

# 4.Problems

## 4.1 Nash Equilibrium

GAN에 비해서 어느정도 완화되긴 했지만 생성기와 분별기의 학습 비율 조정에 어려움이 존재한다.

![Screenshot from 2022-01-17 13-00-35](https://user-images.githubusercontent.com/45096827/149706118-9163a494-baf5-4aa0-8174-c24f470b220e.png)


