# Code base for Fairness constraint to bandit-based online Influence Maximization problem

This is the code reposiratory for paper: "Fairness constraint to bandit-based online Influence
Maximization problem". Paper link is 

## Experiments and previous plots:

alpha is 1.2 in these experiments.

We also tried alpha=0.5 in some case.

Experiments on NBA dataset:

seed size is 10, run 100 iteration,  regret plots can be found in ./presentation/NBA folder.

Initial propagation probability in edge allocated as 0.1.

Experiments on german dataset:

seed size is 10, run 100 iteration,  regret plots can be found in ./presentation/german folder

Initial propagation probability in edge allocated as 0.3.

Experiments on 

seed size is 10, run 100 iteration, regret plots can be found in ./presentation/pokec folder

Initial propagation probability in edge allocated as 0.5.

## Brief explanation of code:

Code reposiratory is based on github repo: 

Bandit repo: [Matrix-Factorization-Bandit/IMFB-KDD2019: Code for the experiments of Matrix Factorization Bandit (github.com)](https://github.com/Matrix-Factorization-Bandit/IMFB-KDD2019)

Offline Oracle repo: [bwilder0/fair_influmax_code_release: Code for the paper &#34;Group-Fairness in Influence Maximization&#34; (github.com)](https://github.com/bwilder0/fair_influmax_code_release)

Migration of offline oracle into online version is the core obstacle.

## Python Environment:

python==3.6, see the requirements.txt

## How to run demo:

for maxmin run FairIM_maxmin.py, for diversity constraint, run FairIM_div.py

for welfare run FairIM_wel.py.(alpha is in Fair_IM_oracle_wel in Fair_Oracle.py, remember to change alpha if trying different cases.)

## Contributors:

Xiangqi Wang, wangxiangqi@mail.ustc.edu.cn
