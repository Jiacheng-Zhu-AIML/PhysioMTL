"""
The quantitative results
"""
from PhysioMTL_solver.PhysioMTL import PhysioMTL
from utils_public_data import get_raw_list_from_public_data, process_for_PhysioMTL_pubdata, \
                            investigate_all, divide_raw_train_test_list, process_for_MTL_pubdata, \
                            k_nearest_model_para_pub, get_pred_Y_test_mtl, investigate_all_model, \
                            investigate_all_model_save

import numpy as np
from matplotlib import pyplot as plt
import pickle


if __name__ == "__main__":
    # First of all... read data
    pkl_file_name = "data_and_pickle/public_data_for_MTL.pkl"

    file_open = open(pkl_file_name, "rb")
    data_dict = pickle.load(file_open)
    file_open.close()

    """
    Draw a good-looking trend.
    remove the outliers and do the learning
    "4": ridiculously high, just remove
    """
    human_feq = 2.0 * np.pi / 24
    subject_id_list = []
    removed_subject_id_list = [4]

    def get_raw_list_from_public_data_custom(data_dict_input, removed_list=removed_subject_id_list):
        key_list = list(data_dict_input.keys())
        t_raw_list = []
        X_raw_list = []
        S_raw_list = []
        Y_raw_list = []
        for key in key_list:
            if key in removed_list:  # 2: non-pattern, 10, 11, non-pattern?
                continue
            t_np, y_np, s_vec = data_dict_input[key]
            sample_num = t_np.shape[0]
            x_raw = np.asarray([np.sin(human_feq * t_np),
                                np.cos(human_feq * t_np),
                                np.ones(sample_num, )]).T

            # Notice: Naive imputation methods
            if key == 18:  # user_18 don't have age data
                s_vec[0] = 22
            if key == 11:  # User_11 does not have sleep data
                s_vec[4] = 6.5  # I use the average
            if key == 3:
                s_vec[5] = 60

            t_raw_list.append(t_np)
            X_raw_list.append(x_raw)
            S_raw_list.append(s_vec)
            Y_raw_list.append(y_np)
            # break
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
    """
    s_vec_base = np.array([[ 25.        ], # Age
                       [1.80        ], # Height (m)
                       [ 75.        ], # weight (kg)
                       [  1.0], # Activity (h)
                       [  9.       ], # Sleep (h)
                       [ 20.        ], # Stress (1)
                       [  1.        ]]).reshape(-1, 1)
    """
    cost_weight_beta = np.array([1, 10.0, 1.0, 10, 1.0, 1.0, 0])


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
                          kernel_sigma=10, T_grad_F_norm_threshold=1e-8)

    PhysioMTL_model.set_aux_feature_cost_function(my_cost_function_pubdata)

    PhysioMTL_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)

    s_vec_base = np.array([[23.],  # Age
                           [1.80],  # Height (m)
                           [85.],  # weight (kg)
                           [1.0],  # Activity (h)
                           [7.],  # Sleep (h)
                           [20.],  # Stress (1)
                           [1.]]).reshape(-1, 1)

    investigate_all_model_save(PhysioMTL_model, s_vec_base)

    plt.show()