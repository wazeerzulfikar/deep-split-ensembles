# Why have a Unified Uncertainty? Disentangling it using Deep Split Ensembles (NeurIPS 2020)

The code is shared for easy reproducibility and to encourage future work.

The following readme has simple steps to reproduce the training, evaluation and all the experiments for any of the datasets (also provided as csv files in supplementary material)

## Setup
1. Setup Virtual Environment
```
pip install virtualenv
virtualenv venv
source venv/bin/activate
```
2. Install dependencies
`pip install -r requirements.txt`

3. Run the code

## Run

### Train
```
python main.py train --datasets_dir datasets --dataset boston --model_dir boston_models
```

### Evaluate
```
python main.py evaluate --datasets_dir datasets --dataset boston --model_dir boston_models
```

### Experiments

#### Calibration - Defer Simulation
```
python main.py experiment --exp_name defer_simulation --plot_name plots --datasets_dir datasets --dataset boston --model_dir boston_models
```

#### Calibration - Clusterwise OOD
```
python main.py experiment --exp_name clusterwise_ood --plot_name plots --datasets_dir datasets --dataset boston --model_dir boston_models
```

#### Calibration - KL Divergence vs Mode
```
python main.py experiment --exp_name kl_mode --plot_name plots --datasets_dir datasets --dataset boston --model_dir boston_models
```

#### Toy regression
```
python main.py experiment --exp_name toy_regression --plot_name toy --model_dir toy_models --dataset toy
```

#### Show model parameters
```
python main.py experiment --exp_name show_summary --datasets_dir datasets --dataset boston
```

#### Empirical rule test
```
python main.py experiment --exp_name empirical_rule_test --datasets_dir datasets --dataset boston
```

## Further Notes

### Human experts

Set `--mod_split` flag in all commands to `human`

### ADReSS - Compare features extraction

* First download the [opensmile](https://www.audeering.com/opensmile/) toolkit.
* Unpack downloaded file using tar -zxvf openSMILE-2.x.x.tar.gz
* Go inside extracted directory cd openSMILE-2.x.x
* Use this command bash autogen.sh or sh autogen.sh
* Use these commands make -j4 ; make
* finally use make install
=======