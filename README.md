# Deep Learning Uncovers Changes in Bleeding Risk Over Time in Patients on Extended Anticoagulation Therapy

### Description
The conventional clinical models used to predict bleeding in patients on anticoagulation therapy are based on a one time baseline measurement. Herein, we demonstration the benefit of using time series follow-up information to improve bleeding prediction and introduce an ensemble of LSTM RNN and feedforward neural network that can use both the baseline and follow-up information to predict bleeding with AUROC of 82% which is 14% higher than the best performing clinical model.

To train the models use the following options `Baseline_Dense`, `LastFUP_Dense`, `FUP_RNN`, or `Ensemble` (Note that Ensemble consists of FUP-RNN and Baseline_Dense, and should be trained last):

```
python train.py [OPTION-NAME]
```
All the follow-up analysis and figures can be performed by:
```
python analyze_results.py
```

# Prerequisites
### Python Environment
[Conda](https://docs.conda.io/) was used to manage python packages and their dependencies. Following commands could be used to install and activate the correct environment:
```
conda env create -f environment.yml
conda activate bleeding-risk-env
```


### Patient Dataset
The raw dataset can be obtained by contacting the corresponding author Dr. Phil Wells (pwells@toh.ca).


