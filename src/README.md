## Implementation notes

The code is based on the PyTorch PPO implementation from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail. This and other requirements are listed in requirements.txt

To launch the described experiments run ```main.py```. To see an example of launching one particular experiment open ```example.ipynb```.

The hyperparameters of PPO are not tuned and listed in ```ppo_config```.

Curiosity modules are implemented in ```curiosity.py```. Scale factors of intrinsic motivations are listed in ```main.py```, they were slightly tuned to produce better results. The main train loop is implemented in ```train.py```.

All intrinsic rewards are simply added to extrinsic reward, no two-headed value network was used. So, to produce better results it may be better to implement it. Another note that may be important is that curiosity module is updated on each step of vector environment, unlike PPO agent which is updated after 128 steps. Updating curiosity module together with PPO agent didn't show any perfomance gain, although it can be investigated more properly.