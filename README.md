# NGSIM-env
This repo is the implementation of the paper "Driving Behavior Modeling using Naturalistic Human Driving Data with Inverse Reinforcement Learning". It contains NGSIM env that can replay vehicle trajectories in the NGSIM dataset while also simulate some interactive behaviors, as well as inverse reinforcement learning (IRL) implementation in this paper for learning driver's reward function.

[**Driving Behavior Modeling using Naturalistic Human Driving Data with Inverse Reinforcement Learning**](https://arxiv.org/abs/2010.03118) 
> Zhiyu Huang, Jingda Wu, Chen Lv        
> IEEE Transactions on Intelligent Transportation Systems

## Getting started
Install the dependent package
```shell
pip install -r requirements.txt
```

Download the NGSIM dataset and put it in the directory.
```shell
git add README.md
git commit -m "Added: README"
git push
```

Run IRL personalized or IRL general.
```shell
git add README.md
git commit -m "Added: README"
git push
```
## Reference
If you find this repo to be useful in your research, please consider citing our work
```
@article{huang2021driving,
  title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
  author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
```

## License
This repo is released under MIT License. The NGSIM data processing code is borrowed from [NGSIM interface](https://github.com/Lemma1/NGSIM-interface). The NGSIM env is built on top of [highway env](https://github.com/eleurent/highway-env) which is released under MIT license.
