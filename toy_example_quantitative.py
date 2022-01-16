from PhysioMTL_solver.PhysioMTL_utils import get_rainbow_curves, \
    get_rainbow_curves_new, process_for_PhysioMTL, get_rainbow_from_s, process_for_MTL, \
    scatter_data_with_s, plot_data_curve_with_s, k_nearest_model_para, compute_list_rmse

from PhysioMTL_solver.PhysioMTL import PhysioMTL

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from mutar import GroupLasso, DirtyModel, MTW
import ot
import pickle


# costumed function for qualitative illustration
# Notice # scatter data samples
def scatter_data_with_s_qua(plt, t_list_raw, Y_raw_list, S_raw_list, rainbow_func=get_rainbow_from_s, **kwargs):
    for task_i, s_value in enumerate(S_raw_list):
        plt.scatter(t_list_raw[task_i], Y_raw_list[task_i], color=rainbow_func(s_value), **kwargs)
        # plt.plot(t_test, pred_Y_list[task_i], label="pred" + str(s_value), color=get_rainbow_from_s(s_value))
    return plt


# Notice # plot
def plot_data_curve_with_s_qua(t_list_or_np, Y_list, S_raw_list, rainbow_func=get_rainbow_from_s, **kwargs):
    if isinstance(t_list_or_np, list):
        for task_i, s_value in enumerate(S_raw_list):
            plt.plot(t_list_or_np[task_i], Y_list[task_i], color=rainbow_func(s_value), **kwargs)


def get_GroupLasso_prediction():
    # Notice: Process data for normal MTL
    X_train_mat, Y_train_mat = process_for_MTL(raw_t_list=t_raw_list, raw_x_list=X_raw_list,
                                               raw_s_list=S_raw_list, raw_y_list=Y_raw_list)

    # Notice # Notice # MTL! # Notice # Notice
    gl = GroupLasso(alpha=0.9)
    gl.fit(X_train_mat, Y_train_mat)  #
    coef = gl.coef_  # Notice: (w_dim = 4, task_num)

    Y_pred = gl.predict(X_train_mat)

    # Notice: For whatever normal MTL method:
    #   Use K-NN to select the similar task and obtain the coef
    pred_list_knn = k_nearest_model_para(Y_pred.T, s_input_list, s_test_list, k=2)
    Y_test_pred = []
    for pred in pred_list_knn:
        Y_test_pred.append(np.mean(pred, axis=1))

    return Y_test_pred


def get_MTW_prediction():
    # # Notice: Process data for normal MTL
    X_train_mat, Y_train_mat = process_for_MTL(raw_t_list=t_raw_list, raw_x_list=X_raw_list,
                                               raw_s_list=S_raw_list, raw_y_list=Y_raw_list)

    # Notice # Notice # MTL! # Notice # Notice
    gl = MTW(alpha=0.9)
    gl.fit(X_train_mat, Y_train_mat)  #
    coef = gl.coef_  # Notice: (w_dim = 4, task_num)

    Y_pred = gl.predict(X_train_mat)

    # Notice: For whatever normal MTL method:
    #   Use K-NN to select the similar task and obtain the coef
    pred_list_knn = k_nearest_model_para(Y_pred.T, s_input_list, s_test_list, k=2)
    Y_test_pred = []
    for pred in pred_list_knn:
        Y_test_pred.append(np.mean(pred, axis=1))

    return Y_test_pred


def get_DirtyModel_prediction():
    # # Notice: Process data for normal MTL
    X_train_mat, Y_train_mat = process_for_MTL(raw_t_list=t_raw_list, raw_x_list=X_raw_list,
                                               raw_s_list=S_raw_list, raw_y_list=Y_raw_list)

    # Notice # Notice # MTL! # Notice # Notice
    gl = DirtyModel(alpha=0.99)
    gl.fit(X_train_mat, Y_train_mat)  #
    coef = gl.coef_  # Notice: (w_dim = 4, task_num)

    Y_pred = gl.predict(X_train_mat)

    # Notice: For whatever normal MTL method:
    #   Use K-NN to select the similar task and obtain the coef
    pred_list_knn = k_nearest_model_para(Y_pred.T, s_input_list, s_test_list, k=2)
    Y_test_pred = []
    for pred in pred_list_knn:
        Y_test_pred.append(np.mean(pred, axis=1))

    return Y_test_pred


if __name__ == "__main__":

    # Notice: This is the underlying function that controls the curves
    def underlying_truth(s):
        A = 0.2 * s + 3.0
        phi = 0.4 * s - 1.8
        M = 5 * s + 35
        return A, phi, M

    # Notice: generate data for training
    s_input_list = [1, 2, 3, 4, 5, 6]


    freq = 0.34

    t_raw_list, X_raw_list, S_raw_list, Y_raw_list = get_rainbow_curves_new(s_input_list,
                                                                            data_noise=0.5,
                                                                            underlying_func=underlying_truth)
    # Notice: Process data for PhysioMTL
    t_list, X_train_list, S_train_list, Y_train_list = process_for_PhysioMTL(raw_t_list=t_raw_list,
                                                                          raw_x_list=X_raw_list,
                                                                          raw_s_list=S_raw_list,
                                                                          raw_y_list=Y_raw_list)

    # Notice: generate data for testing
    s_test_list = np.random.uniform(low=4, high=9.9, size=5).tolist()

    t_test_list_raw, X_test_raw_list, S_test_raw_list, Y_test_raw_list = get_rainbow_curves_new(s_test_list,
                                                                            data_noise=0.5,
                                                                            underlying_func=underlying_truth)
    # Notice: Process data for PhysioMTL
    t_test_list, X_test_list, S_test_list, Y_test_list_groundtruth = process_for_PhysioMTL(raw_t_list=t_test_list_raw,
                                                                          raw_x_list=X_test_raw_list,
                                                                          raw_s_list=S_test_raw_list,
                                                                          raw_y_list=Y_test_raw_list)

    # Notice: PhysioMTL
    def my_cost_function(x, y):
        return np.sqrt(np.mean(np.square(x - y)))

    PhysioMTL_model = PhysioMTL(alpha=0.1, T_initial=None,
                          T_lr=9e-2, W_lr=1e-3, T_ite_num=100, W_ite_num=100,
                          all_ite_num=50, map_type="linear", kernel_sigma=30)
    PhysioMTL_model.set_aux_feature_cost_function(my_cost_function)

    PhysioMTL_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)
    W_opt, T_opt = PhysioMTL_model.coef_

    # Notice: The results on training set
    pred_f_train_list = PhysioMTL_model.predict()
    # Notice: The results on testing set
    pred_f_test_list = PhysioMTL_model.predict(X_list=X_test_list, S_list=S_test_list)
    PhysioMTL_rmse = compute_list_rmse(list_a=pred_f_test_list, list_b=Y_test_raw_list)

    # Notice: Other Baseline methods
    print()

    Y_pred_grouplasso = get_GroupLasso_prediction()
    grouplasso_mse = compute_list_rmse(list_a=Y_pred_grouplasso, list_b=Y_test_raw_list)
    print("grouplasso_mse =", grouplasso_mse)

    Y_pred_DirtyModel = get_DirtyModel_prediction()
    DirtyModel_mse = compute_list_rmse(list_a=Y_pred_DirtyModel, list_b=Y_test_raw_list)
    print("DirtyModel_mse =", DirtyModel_mse)

    Y_pred_MTW = get_MTW_prediction()
    MTW_mse = compute_list_rmse(list_a=Y_pred_MTW, list_b=Y_test_raw_list)
    print("MTW_mse =", MTW_mse)

    print()
    print("PhysioMTL_rmse =", PhysioMTL_rmse)

    # Notice: Get the W-distance
    src_np = np.array(s_input_list).reshape((-1, 1))
    tgt_np = np.array(s_test_list).reshape((-1, 1))
    w_distance = ot.wasserstein_1d(src_np, tgt_np)

    print()
    print("w_distance =", w_distance)

    # Notice: Plot the training data and training result
    fig_c = plt.figure(666, figsize=(8, 6))
    scatter_data_with_s_qua(plt, t_raw_list, Y_raw_list, S_raw_list, rainbow_func=get_rainbow_from_s)
    plot_data_curve_with_s_qua(t_raw_list, pred_f_train_list, S_raw_list, rainbow_func=get_rainbow_from_s)
    colors_list_train = ["grey", "grey"]
    lines_list_train = [Line2D([0], [0], color="grey", linewidth=0, linestyle="solid", marker="o", markersize=10),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="s", markersize=2)]
    labels_list_train = ["data", "regression"]
    plt.legend(lines_list_train, labels_list_train)
    plt.title("Training tasks")
    plt.xlim(-2, 26)
    plt.ylim(30, 100)

    # Notice: Generalization
    # Notice: step 1, plot the test data
    fig_test = plt.figure(233, figsize=(8, 6))
    scatter_data_with_s_qua(plt, t_test_list_raw, Y_test_raw_list, S_test_raw_list,
                            rainbow_func=get_rainbow_from_s, alpha=0.4, s=60)
    plot_data_curve_with_s_qua(t_test_list_raw, pred_f_test_list, S_test_raw_list,
                               rainbow_func=get_rainbow_from_s,
                               linewidth=3, linestyle="solid", marker="s", markersize=2)
    plot_data_curve_with_s_qua(t_test_list_raw, Y_pred_grouplasso, S_test_raw_list,
                               rainbow_func=get_rainbow_from_s,
                               linewidth=3, linestyle="dashed", marker="^", markersize=2)
    plot_data_curve_with_s_qua(t_test_list_raw, Y_pred_MTW, S_test_raw_list,
                               rainbow_func=get_rainbow_from_s,
                               linewidth=3, linestyle="dotted", marker="^", markersize=2)
    colors_list = ["grey", "grey", "grey", "grey"]
    lines_list = [Line2D([0], [0], color="grey", linewidth=0, linestyle="solid", marker="o", markersize=10),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="s", markersize=2),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="dashed", marker="^", markersize=2),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="dotted", marker="^", markersize=2)]
    labels_list = ["data", "PhysioMTL", "Grouplasso", "MTL"]

    plt.legend(lines_list, labels_list)
    plt.title("Adapt to unseen tasks")
    plt.xlim(-2, 26)
    plt.ylim(30, 100)
    plt.show()