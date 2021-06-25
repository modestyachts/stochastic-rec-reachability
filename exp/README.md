## Experiments folder

Upon running reachability audits the `exp` folder will get populated with files corresponding to experimental runs.

The current folder structure for experimental results is as follows:

```
stochastic-rec-reachability
├─ exp
│  ├─ ml-100k_libfm <----- {dataset}_{recommender} folder that contains all runs
│  │  ├─ test_2021-01-04-18:52  <----- {titlestr}_{datestr} folder that corresponds to a run
│  │  │ └─ params.pkl <----- contains run_params such as betas, action parameters (etc)
│  │  │  ├─ action_5_target_1_beta_1 <----- folder corresponding to a full experiment
│  │  │  │  ├─ experiment.pkl <----- general exp_file data such as target_mask, action_mask, etc
│  │  │  │  ├─ user_0.pkl  <----- dict of max_rho, baseline_rho, etc of one user
│  │  │  │  ├─ user_1.pkl
│  │  │  │  ├─ user_2.pkl
│  │  │  │  ├─ user_3.pkl
│  │  │  │  ├─ ...
│  │     ├─ k_5_target_1_beta_2 <----- folder corresponding to another experiment in the same run
│  │     │  ├─ experiment.pkl
│  │     │  ├─ user_0.pkl
│  │     │  ├─ user_1.pkl
│  │  │  │  ├─ ...
```