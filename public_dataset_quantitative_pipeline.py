"""
Run PhysioMTL on the processed MMASH dataset.
Compare the PhysioMTL method with other baselines quantitatively
in cross-validation fashion.
"""
import pickle

import numpy as np
from mutar import GroupLasso, DirtyModel, MTW, IndLasso, MultiLevelLasso, ReMTW

from PhysioMTL_solver.PhysioMTL import PhysioMTL
from PhysioMTL_solver.PhysioMTL_utils import compute_list_rmse
from utils_public_data import process_for_PhysioMTL_pubdata, \
    divide_raw_train_test_list, process_for_MTL_pubdata, \
    get_baseline_MTL_mse

# Notice: The function to get raw data
human_feq = 2.0 * np.pi / 24
removed_subject_id_list = [4]


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
    subject_id_list = []
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

        subject_id_list.append(key)
    return t_raw_list, X_raw_list, S_raw_list, Y_raw_list, subject_id_list


if __name__ == "__main__":

    # Notice: Get the processed data
    pkl_file_name = "data_and_pickle/public_data_for_MTL.pkl"

    file_open = open(pkl_file_name, "rb")
    data_dict = pickle.load(file_open)
    file_open.close()

    t_raw_list, X_raw_list, S_raw_list, Y_raw_list, subject_id_test_list = get_raw_list_from_public_data_custom(
        data_dict)

    training_ratio = 0.6

    test_alpha = 0.9  # Notice: For baseline

    print("training_ratio = ", training_ratio, "  test_alpha = ", test_alpha)

    PhysioMTL_kernel_mse_list = []
    PhysioMTL_mse_list = []
    indlasso_mse_list = []
    grouplasso_mse_list = []
    mllasso_mse_list = []
    dirty_mse_list = []
    mtw_mse_list = []
    remtw_mse_list = []

    mean_mse_list = []

    experiment_time = 10
    for exp_id in range(experiment_time):
        # Notice: Divide data into training and testing set
        raw_train_tuple, raw_test_tuple = divide_raw_train_test_list(t_raw_list_input=t_raw_list,
                                                                     X_raw_list_input=X_raw_list,
                                                                     S_raw_list_input=S_raw_list,
                                                                     Y_raw_list_input=Y_raw_list,
                                                                     tr_ratio=training_ratio)
        # Notice: For PhysioMTL
        t_train_list, X_train_list, S_train_list, Y_train_list = process_for_PhysioMTL_pubdata(
            raw_t_list=raw_train_tuple[0],
            raw_x_list=raw_train_tuple[1],
            raw_s_list=raw_train_tuple[2],
            raw_y_list=raw_train_tuple[3])
        t_test_list, X_test_list, S_test_list, Y_test_list = process_for_PhysioMTL_pubdata(raw_t_list=raw_test_tuple[0],
                                                                                           raw_x_list=raw_test_tuple[1],
                                                                                           raw_s_list=raw_test_tuple[2],
                                                                                           raw_y_list=raw_test_tuple[3])

        cost_weight_beta = np.array([1, 10.0, 1.0, 10, 1.0, 1.0, 0])


        def my_cost_function_pubdata(x_vec, y_vec):
            weighted_diff = np.dot(cost_weight_beta, x_vec - y_vec)
            return np.sqrt(np.mean(np.square(weighted_diff)))


        # Notice: PhysioMTL Linear
        PhysioMTL_model = PhysioMTL(alpha=0.1, T_initial=None,
                                    T_lr=1e-3, W_lr=1e-7,
                                    T_ite_num=200, W_ite_num=50,
                                    all_ite_num=50,
                                    verbose_T_grad=False)

        PhysioMTL_model.set_aux_feature_cost_function(my_cost_function_pubdata)
        PhysioMTL_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)
        pred_Y_test_list = PhysioMTL_model.predict(X_list=X_test_list, S_list=S_test_list)
        PhysioMTL_mse = compute_list_rmse(pred_Y_test_list, Y_test_list)

        # Notice: PhysioMTL_kernel
        PhysioMTL_kernel_model = PhysioMTL(alpha=0.1, T_initial=None,
                                           T_lr=9e-2, W_lr=1e-7,
                                           T_ite_num=200, W_ite_num=50,
                                           all_ite_num=50,
                                           verbose_T_grad=False,
                                           map_type="kernel", kernel_cost_function=my_cost_function_pubdata,
                                           kernel_sigma=40)

        PhysioMTL_kernel_model.set_aux_feature_cost_function(my_cost_function_pubdata)
        PhysioMTL_kernel_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)
        pred_Y_kernel_test_list = PhysioMTL_kernel_model.predict(X_list=X_test_list, S_list=S_test_list)
        PhysioMTL_kernel_mse = compute_list_rmse(pred_Y_kernel_test_list, Y_test_list)

        # Notice: baseline methods
        X_train_mat, Y_train_mat = process_for_MTL_pubdata(raw_t_list=raw_train_tuple[0],
                                                           raw_x_list=raw_train_tuple[1],
                                                           raw_s_list=raw_train_tuple[2],
                                                           raw_y_list=raw_train_tuple[3])

        # Define the kernel function for generalization test
        cost_weight_beta_mtl = np.array([1, 10.0, 1.0, 10, 1.0, 1.0])


        def my_cost_function_pubdata_pub(x_vec, y_vec):
            weighted_diff = np.dot(cost_weight_beta_mtl, x_vec - y_vec)
            return np.sqrt(np.mean(np.square(weighted_diff)))


        mtl_grouplasso = GroupLasso(alpha=test_alpha)
        grouplasso_mse = get_baseline_MTL_mse(mtl_grouplasso, X_train_mat, Y_train_mat, raw_train_tuple,
                                              raw_test_tuple, my_cost_function_pubdata_pub)

        mtl_indlasso = IndLasso(alpha=test_alpha * np.ones((X_train_mat.shape[0])))
        indlasso_mse = get_baseline_MTL_mse(mtl_indlasso, X_train_mat, Y_train_mat, raw_train_tuple,
                                            raw_test_tuple, my_cost_function_pubdata_pub)

        mtl_mllasso = MultiLevelLasso(alpha=test_alpha)
        mllasso_mse = get_baseline_MTL_mse(mtl_mllasso, X_train_mat, Y_train_mat, raw_train_tuple,
                                           raw_test_tuple, my_cost_function_pubdata_pub)

        mtl_dirty = DirtyModel(alpha=test_alpha)
        dirty_mse = get_baseline_MTL_mse(mtl_dirty, X_train_mat, Y_train_mat, raw_train_tuple,
                                         raw_test_tuple, my_cost_function_pubdata_pub)

        mtl_mtw = MTW(alpha=test_alpha)
        mtw_mse = get_baseline_MTL_mse(mtl_mtw, X_train_mat, Y_train_mat, raw_train_tuple,
                                       raw_test_tuple, my_cost_function_pubdata_pub)

        mtl_remtw = ReMTW(alpha=test_alpha)
        remtw_mse = get_baseline_MTL_mse(mtl_remtw, X_train_mat, Y_train_mat, raw_train_tuple,
                                         raw_test_tuple, my_cost_function_pubdata_pub)


        def get_global_average_mse(X_train_mat, Y_train_mat, raw_train_tuple, raw_test_tuple):
            mean_on_train = np.mean(Y_train_mat)
            mean_pred_list = []
            for array_i in raw_test_tuple[3]:
                mean_pred_list.append(mean_on_train * np.ones_like(array_i))
            mse = compute_list_rmse(mean_pred_list, raw_test_tuple[3])
            return mse


        mean_mse = get_global_average_mse(X_train_mat, Y_train_mat, raw_train_tuple, raw_test_tuple)

        print("PhysioMTL_kernel_mse =", PhysioMTL_kernel_mse)
        print("PhysioMTL_mse =", PhysioMTL_mse)
        print("indlasso_mse =", indlasso_mse)
        print("grouplasso_mse =", grouplasso_mse)
        print("mllasso_mse =", mllasso_mse)
        print("dirty_mse =", dirty_mse)
        print("mtw_mse =", mtw_mse)
        print("remtw_mse =", remtw_mse)
        print("mean_mse =", mean_mse)
        print()

        PhysioMTL_kernel_mse_list.append(PhysioMTL_kernel_mse)
        PhysioMTL_mse_list.append(PhysioMTL_mse)
        indlasso_mse_list.append(indlasso_mse)
        grouplasso_mse_list.append(grouplasso_mse)
        mllasso_mse_list.append(mllasso_mse)
        dirty_mse_list.append(dirty_mse)
        mtw_mse_list.append(mtw_mse)
        remtw_mse_list.append(remtw_mse)
        mean_mse_list.append(mean_mse)

    # Notice: Done
    print()
    print("At the end of the day")
    print()
    print("training_ratio = ", training_ratio, "  test_alpha = ", test_alpha)
    print()

    print("global mean: mean = ", np.mean(mean_mse_list), " std = ", np.std(mean_mse_list))
    print("indlasso: mean = ", np.mean(indlasso_mse_list), " std = ", np.std(indlasso_mse_list))
    print("grouplasso: mean = ", np.mean(grouplasso_mse_list), " std = ", np.std(grouplasso_mse_list))
    print("mllasso: mean = ", np.mean(mllasso_mse_list), " std = ", np.std(mllasso_mse_list))
    print("dirty: mean = ", np.mean(dirty_mse_list), " std = ", np.std(dirty_mse_list))
    print("mtw: mean = ", np.mean(mtw_mse_list), " std = ", np.std(mtw_mse_list))
    print("remtw: mean = ", np.mean(remtw_mse_list), " std = ", np.std(remtw_mse_list))

    print("PhysioMTL: mean = ", np.mean(PhysioMTL_mse_list), " std = ", np.std(PhysioMTL_mse_list))
    print("PhysioMTL_kernel: mean = ", np.mean(PhysioMTL_kernel_mse_list), " std = ", np.std(PhysioMTL_kernel_mse_list))
