# PhysioMTL


## Requirement
* Python 3
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

* To reproduce the results, e.g. counterfactual analysis: 
```
python public_dataset_counterfactual_plots_nonlinear.py
```
* To use the PhysioMTL package on your own dataset:
```
from PhysioMTL_solver.PhysioMTL import PhysioMTL

def my_cost_function_pubdata(x_vec, y_vec):
    weighted_diff = np.dot(cost_weight_beta, x_vec - y_vec)
    return np.sqrt(np.mean(np.square(weighted_diff)))
    
PhysioMTL_model = PhysioMTL(alpha=0.1, kernel_cost_function=my_cost_function_pubdata,
                          kernel_sigma=10, T_grad_F_norm_threshold=1e-8)

PhysioMTL_model.set_aux_feature_cost_function(my_cost_function_pubdata)

PhysioMTL_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)
```