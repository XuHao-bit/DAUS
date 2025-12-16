# TDAS
Code of the paper "[TOIS 25] Dual-Adaptive Update Strategies Enhanced Meta-Optimization for User Cold-Start Recommendation", which is an journal extension of conference paper "[CIKM 2023] Task-Difficulty-Aware Meta-Learning with Adaptive Update Strategies for User Cold-Start Recommendation".
We have the following major extensions:
- We propose a dual-adaptive update strategies enhanced meta-optimization framework, integrating both adaptive hyperparameters and objectives. In contrast, TDAS is only single-adaptive to hyperparameters. 
- We additionally employ SSL in our multifaceted task encoder to capture implicit task information. In contrast, TDAS relies solely on manually designed explicit task features. 
- We conduct more experiments including robust experiment, indepth study, etc, to valid the effectiveness of proposed dual-adaptive update strategies and SSL-based task encoder.

## Requirements 
- Python 3.7
- pytorch 1.1.0
- numpy 1.21.5
- pandas 1.3.5

## Dataset

1. The data preprocessing is following our conference version: [TDAS [CIKM'23] Data Preparing](https://github.com/XuHao-bit/TDAS/tree/main/prepare_data)
2. Our code for data processing are implemented based on the code of [MAMO [KDD'20]](https://github.com/dongmanqing/Code-for-MAMO)

## Model training
The structure of our code: 
```
- prepare_data
  prepareList.py
  prepareMovielens.py
- modules
  BaseRecModel.py: Implementation of the base recommender.
  HyperParamModel.py: Implementation of the task adaptive hyperparameter generator.
  MyOptim.py: Implementation of the Adam optimizer (for global updating the base model).
  TaskEncoder.py: Implementation of the task difficulty encoder.
- robust_loss
  This part is implemented based on the code of https://github.com/jonbarron/robust_loss_pytorch.
DataLoading.py
prepareDataset.py
run_experiment.py
run_robust_experiment.py
TaskDiffModel.py
Trainer.py
utils.py
```

## Start
Change your running config at `run_experiment.py` and `utils.py`, and start a training procedure by:
```
python run_experiment.py --mode tdmeta >> [result path]
```

## Citation 
If you use this code, please consider to cite the following paper:

```
@article{DBLP:journals/tois/ZhaoZWJMYT25,
  author       = {Xuhao Zhao and
                  Yanmin Zhu and
                  Chunyang Wang and
                  Mengyuan Jing and
                  Wenze Ma and
                  Jiadi Yu and
                  Feilong Tang},
  title        = {Dual-Adaptive Update Strategies-Enhanced Meta-Optimization for User
                  Cold-Start Recommendation},
  journal      = {{ACM} Trans. Inf. Syst.},
  volume       = {43},
  number       = {6},
  pages        = {154:1--154:36},
  year         = {2025}
}
```
```
@inproceedings{DBLP:conf/cikm/ZhaoZWJYT23,
  author       = {Xuhao Zhao and
                  Yanmin Zhu and
                  Chunyang Wang and
                  Mengyuan Jing and
                  Jiadi Yu and
                  Feilong Tang},
  title        = {Task-Difficulty-Aware Meta-Learning with Adaptive Update Strategies
                  for User Cold-Start Recommendation},
  booktitle    = {{CIKM}},
  pages        = {3484--3493},
  publisher    = {{ACM}},
  year         = {2023}
}
```
