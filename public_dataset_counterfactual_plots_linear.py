"""
Run PhysioMTL on the processed MMASH dataset.
Produce the counterfactual analysis.
"""
import pickle

import numpy as np
from matplotlib import pyplot as plt

from PhysioMTL_solver.PhysioMTL import PhysioMTL
from utils_public_data import process_for_PhysioMTL_pubdata, \
    investigate_all_model_save_linear

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


    def get_raw_list_from_public_data_custom(data_dict_input, removed_list=removed_subject_id_list):
        """
        Convert the preprocessed MMASH data for multitask regression models.
        Remove outliers and do imputation.
        """
        key_list = list(data_dict_input.keys())
        t_raw_list = []
        X_raw_list = []
        S_raw_list = []
        Y_raw_list = []
        for key in key_list:
            if key in removed_list:
                continue
            t_np, y_np, s_vec = data_dict_input[key]
            sample_num = t_np.shape[0]
            x_raw = np.asarray([np.sin(human_feq * t_np),
                                np.cos(human_feq * t_np),
                                np.ones(sample_num, )]).T

            # Notice: Do simple imputation.
            if key == 18:  # user_18 don't have age data
                s_vec[0] = 16
            if key == 11:  # User_11 does not have sleep data
                s_vec[4] = 6.0  # I use the average
            if key == 3:
                s_vec[5] = 60

            t_raw_list.append(t_np)
            X_raw_list.append(x_raw)
            S_raw_list.append(s_vec)
            Y_raw_list.append(y_np)
            subject_id_list.append(key)
        return t_raw_list, X_raw_list, S_raw_list, Y_raw_list, subject_id_list


    # Notice: process data
    t_raw_list, X_raw_list, S_raw_list, Y_raw_list, subject_id_test_list = get_raw_list_from_public_data_custom(
        data_dict)

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
                                T_lr=1.1e-3, W_lr=1e-7,
                                T_ite_num=200, W_ite_num=50,
                                all_ite_num=10,
                                verbose_T_grad=True,
                                map_type="linear", kernel_cost_function=my_cost_function_pubdata,
                                kernel_sigma=10, T_grad_F_norm_threshold=1e-7)

    PhysioMTL_model.set_aux_feature_cost_function(my_cost_function_pubdata)

    PhysioMTL_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)

    """
    s_vec_base = np.array([[ 25.        ], # Age
                       [1.80        ], # Height (m)
                       [ 75.        ], # weight (kg)
                       [  1.0], # Activity (h)
                       [  9.       ], # Sleep (h)
                       [ 20.        ], # Stress (1)
                       [  1.        ]]).reshape(-1, 1)
    """
    s_vec_base = np.array([[23.],  # Age
                           [1.80],  # Height (m)
                           [85.],  # weight (kg)
                           [1.0],  # Activity (h)
                           [7.],  # Sleep (h)
                           [20.],  # Stress (1)
                           [1.]]).reshape(-1, 1)

    investigate_all_model_save_linear(PhysioMTL_model, s_vec_base)

    plt.show()
