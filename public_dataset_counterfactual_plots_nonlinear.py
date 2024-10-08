"""
Run PhysioMTL on the processed MMASH dataset.
Produce the counterfactual analysis with the nonlinear kernel map.
"""
import pickle

import numpy as np
from matplotlib import pyplot as plt

from PhysioMTL_solver.PhysioMTL import PhysioMTL
from utils_public_data import process_for_PhysioMTL_pubdata, \
    investigate_all_model_save,  get_raw_list_from_public_data_custom

if __name__ == "__main__":
    # Notice: First of all... read data
    pkl_file_name = "data_and_pickle/public_data_for_MTL.pkl"

    file_open = open(pkl_file_name, "rb")
    data_dict = pickle.load(file_open)
    file_open.close()

    """
    remove the outliers 
    "4": ridiculously high, just remove.
    "8": age=40, stress>70, but has very good HRV. 
         Could be an athlete.
    """
    human_feq = 2.0 * np.pi / 24
    subject_id_list = []
    removed_subject_id_list = [4, 8]

    # Notice: process data
    t_raw_list, X_raw_list, S_raw_list, Y_raw_list, subject_id_test_list = get_raw_list_from_public_data_custom(
        data_dict, removed_subject_id_list)

    t_list, X_train_list, S_train_list, Y_train_list = process_for_PhysioMTL_pubdata(raw_t_list=t_raw_list,
                                                                                     raw_x_list=X_raw_list,
                                                                                     raw_s_list=S_raw_list,
                                                                                     raw_y_list=Y_raw_list)

    # Notice: define a cost metric
    cost_weight_beta = np.array([1, 10.0, 1.0, 10, 1.0, 1.0, 0])

    # Notice: Define the cost function here
    def my_cost_function_pubdata(x_vec, y_vec):
        weighted_diff = np.dot(cost_weight_beta, x_vec - y_vec)
        return np.sqrt(np.mean(np.square(weighted_diff)))


    T_ini = None

    PhysioMTL_model = PhysioMTL(alpha=0.1, T_initial=T_ini,
                                T_lr=9e-2, W_lr=1e-7,
                                T_ite_num=200, W_ite_num=50,
                                all_ite_num=50,
                                verbose_T_grad=True,
                                map_type="kernel", kernel_cost_function=my_cost_function_pubdata,
                                kernel_sigma=10, T_grad_F_norm_threshold=1e-9)

    PhysioMTL_model.set_aux_feature_cost_function(my_cost_function_pubdata)

    PhysioMTL_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)

    s_vec_base = np.array([[25.],  # Age
                           [1.80],  # Height (m)
                           [85.],  # weight (kg)
                           [1.0],  # Activity (h)
                           [7.],  # Sleep (h)
                           [20.],  # Stress (1)
                           [1.]]).reshape(-1, 1)

    investigate_all_model_save(PhysioMTL_model, s_vec_base)

    plt.show()
