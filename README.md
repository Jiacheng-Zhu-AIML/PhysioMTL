## PhysioMTL: Personalizing Physiological Patterns using Optimal Transport Multi-Task Regression


This code accompanies the research paper:

[PhysioMTL: Personalizing Physiological Patterns using Optimal Transport Multi-Task Regression](https://proceedings.mlr.press/v174/zhu22a.html)

Conference on Health, Inference, and Learning (CHIL) 2022, Proceedings of Machine Learning Research (PMLR)

[[PMLR]](https://proceedings.mlr.press/v174/zhu22a.html), [[arXiv]](https://arxiv.org/abs/2203.12595), [[talk]](https://slideslive.com/38979164/physiomtl-personalizing-physiological-patterns-using-optimal-transport-multitask-regression?ref=speaker-23217)

## Introduction

Heart rate variability (HRV) is a practical and noninvasive measure of autonomic nervous system activity, which plays an essential role in cardiovascular health. However, using HRV to assess physiology status is challenging. Even in clinical settings, HRV is sensitive to acute stressors such as physical activity, mental stress, hydration, alcohol, and sleep. Wearable devices provide convenient HRV measurements, but the irregularity of measurements and uncaptured stressors can bias conventional analytical methods. To better interpret HRV measurements for downstream healthcare applications, we learn a personalized diurnal rhythm as an accurate physiological indicator for each individual. We develop Physiological Multitask-Learning (PhysioMTL) by harnessing Optimal Transport theory within a Multitask-learning (MTL) framework. The proposed method learns an individual-specific predictive model from heterogeneous observations, and enables estimation of an optimal transport map that yields a push forward operation onto the demographic features for each task.

The code here is a basic illustration of the implementation on the publicly available [MMASH dataset](https://physionet.org/content/mmash/1.0.0/). 

## Requirement
* Python 3.7
* Numpy
* Mutar package [[link]](https://github.com/hichamjanati/mutar)
* MMASH dataset (optional) [[link]](https://physionet.org/content/mmash/1.0.0/)

## File Description
* ``PhysioMTL_solver`` includes the solver and utils functions for PhysioMTL model
* ``toy_example_qualitative.py`` and ``toy_example_quantitative.py`` provide all results on a simulated dataset.
* ``public_dataset_counterfactual_*.py`` provide the counterfactual analysis results on the MMASH dataset.
* ``public_dataset_quantitative_pipeline.py`` conducts numerical experiments on the MMASH dataset.
* ``data_and_pickle/public_data_for_MTL.pkl`` stores the pre-processed data from MMASH dataset 

## Usage

* Process and MMASH dataset for the HRV values:
```
python public_dataset_preprocessing.py
```

* To reproduce the results, e.g. counterfactual analysis: 
```
python public_dataset_counterfactual_plots_nonlinear.py
```
* To use the PhysioMTL package on your own dataset:
```python 
from PhysioMTL_solver.PhysioMTL import PhysioMTL

# cost_weight_beta is the self-defined coefficient for the ground metric for task (individual) similarity
def my_cost_function_pubdata(x_vec, y_vec):
    weighted_diff = np.dot(cost_weight_beta, x_vec - y_vec)
    return np.sqrt(np.mean(np.square(weighted_diff)))

# Initialize the solver    
PhysioMTL_model = PhysioMTL(alpha=0.1, kernel_cost_function=my_cost_function_pubdata,
                          kernel_sigma=10, T_grad_F_norm_threshold=1e-8)

# Use the self-defined ground metric 
PhysioMTL_model.set_aux_feature_cost_function(my_cost_function_pubdata)

# Solve it
PhysioMTL_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)
```