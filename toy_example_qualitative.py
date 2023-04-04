"""
Generate a synthetic sinusoidal dataset.
Run PhysioMTL and mutar multitask regression baselines on the synthetic dataset.
Compare the test error RMSE in a cross-validation fashion.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mutar import GroupLasso, DirtyModel, MTW

from PhysioMTL_solver.PhysioMTL import PhysioMTL
from PhysioMTL_solver.PhysioMTL_utils import get_rainbow_curves_new, process_for_PhysioMTL, get_rainbow_from_s, \
    process_for_MTL, k_nearest_model_para, scatter_data_with_s_qua, plot_data_curve_with_s_qua


def get_GroupLasso_prediction():
    """
    Trains a mutar.GroupLasso model and makes predictions on the test set.

    Returns:
        list: A list of predicted outputs for the test set.
        
    """
    # Process data for normal MTL
    X_train_mat, Y_train_mat = process_for_MTL(raw_t_list=t_raw_list, raw_x_list=X_raw_list,
                                               raw_s_list=S_raw_list, raw_y_list=Y_raw_list)

    # Do the MTL training
    gl = GroupLasso(alpha=0.9)
    gl.fit(X_train_mat, Y_train_mat)

    Y_pred = gl.predict(X_train_mat)

    #  Use K-NN to select the similar task and obtain the coef
    pred_list_knn = k_nearest_model_para(Y_pred.T, s_input_list, s_test_list, k=2)
    Y_test_pred = []
    for pred in pred_list_knn:
        Y_test_pred.append(np.mean(pred, axis=1))

    return Y_test_pred


def get_MTW_prediction():
    """
    Trains a mutar.MTW model (Multi-task Wasserstein [1]) and makes predictions on the test set.
    [1] Wasserstein regularization for sparse multi-task regression, Janati et al., AISTATS 2019.

    Returns:
        list: A list of predicted outputs for the test set.
        
    """
    # Process data for normal MTL
    X_train_mat, Y_train_mat = process_for_MTL(raw_t_list=t_raw_list, raw_x_list=X_raw_list,
                                               raw_s_list=S_raw_list, raw_y_list=Y_raw_list)

    # Do the MTL training
    gl = MTW(alpha=0.9)
    gl.fit(X_train_mat, Y_train_mat)

    Y_pred = gl.predict(X_train_mat)

    #  Use K-NN to select the similar task and obtain the coef
    pred_list_knn = k_nearest_model_para(Y_pred.T, s_input_list, s_test_list, k=2)
    Y_test_pred = []
    for pred in pred_list_knn:
        Y_test_pred.append(np.mean(pred, axis=1))

    return Y_test_pred


def get_DirtyModel_prediction():
    """
    Trains a mutar.DirtyModel model and makes predictions on the test set.

    Returns:
        list: A list of predicted outputs for the test set.
        
    """
    # # Notice: Process data for normal MTL
    X_train_mat, Y_train_mat = process_for_MTL(raw_t_list=t_raw_list, raw_x_list=X_raw_list,
                                               raw_s_list=S_raw_list, raw_y_list=Y_raw_list)

    # Do the MTL training
    gl = DirtyModel(alpha=0.99)
    gl.fit(X_train_mat, Y_train_mat)

    Y_pred = gl.predict(X_train_mat)

    #  Use K-NN to select the similar task and obtain the coef
    pred_list_knn = k_nearest_model_para(Y_pred.T, s_input_list, s_test_list, k=2)
    Y_test_pred = []
    for pred in pred_list_knn:
        Y_test_pred.append(np.mean(pred, axis=1))

    return Y_test_pred


if __name__ == "__main__":
    # Notice, generate data
    #   This is the underlying function that decides the curves
    def underlying_truth(s):
        A = 0.2 * s + 3.0
        phi = 0.4 * s - 1.8
        M = 5 * s + 35
        return A, phi, M


    # Notice: generate data for training
    s_input_list = [1, 2, 2.2, 4, 5, 6, 6.8]
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
    s_test_list = [0.1, 3, 3.5, 7.5, 8.5, 8.0]
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
                                T_lr=5e-2, W_lr=1e-3, T_ite_num=100, W_ite_num=100,
                                all_ite_num=50)
    PhysioMTL_model.set_aux_feature_cost_function(my_cost_function)

    PhysioMTL_model.fit(X_list=X_train_list, Y_list=Y_train_list, S_list=S_train_list)
    W_opt, T_opt = PhysioMTL_model.coef_

    print("W_opt =")
    print(W_opt)
    print("T_opt = ")
    print(T_opt)

    # Notice: show the results on training set
    pred_f_train_list = PhysioMTL_model.predict()
    # Notice: show the results on testing set
    pred_f_test_list = PhysioMTL_model.predict(X_list=X_test_list, S_list=S_test_list)

    # Notice: Other Baseline methods
    Y_pred_grouplasso = get_GroupLasso_prediction()
    Y_pred_MTW = get_MTW_prediction()
    Y_pred_DirtyModel = get_DirtyModel_prediction()

    # Notice: Plot the training data and training result
    fig_c = plt.figure(666, figsize=(8, 6), dpi=300)
    scatter_data_with_s_qua(plt, t_raw_list, Y_raw_list, S_raw_list, rainbow_func=get_rainbow_from_s, s=60)
    colors_list_train = ["grey", "grey"]
    lines_list_train = [Line2D([0], [0], color="grey", linewidth=0, linestyle="solid", marker="o", markersize=10),
                        ]
    labels_list_train = ["data", ]
    plt.legend(lines_list_train, labels_list_train, loc="upper right", fontsize=16)
    plt.title("Training tasks", fontsize=16)
    plt.xlabel("x", fontsize=18, labelpad=-5)
    plt.ylabel("f(x)", fontsize=18)
    plt.xlim(-1, 27)
    plt.ylim(30, 90)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(pad=0)

    # Notice: Test the generalization and plot the test data
    fig_test = plt.figure(233, figsize=(8, 6), dpi=300)
    scatter_data_with_s_qua(plt, t_test_list_raw, Y_test_raw_list, S_test_raw_list,
                            rainbow_func=get_rainbow_from_s, alpha=0.6, s=60)
    plot_data_curve_with_s_qua(t_test_list_raw, pred_f_test_list, S_test_raw_list,
                               rainbow_func=get_rainbow_from_s,
                               linewidth=5, linestyle="solid", marker="s", markersize=2)
    plot_data_curve_with_s_qua(t_test_list_raw, Y_pred_grouplasso, S_test_raw_list,
                               rainbow_func=get_rainbow_from_s,
                               linewidth=3, linestyle="dashed", marker="^", markersize=2, alpha=0.5)
    plot_data_curve_with_s_qua(t_test_list_raw, Y_pred_MTW, S_test_raw_list,
                               rainbow_func=get_rainbow_from_s,
                               linewidth=3, linestyle="dotted", marker="^", markersize=2, alpha=0.5)
    plot_data_curve_with_s_qua(t_test_list_raw, Y_pred_DirtyModel, S_test_raw_list,
                               rainbow_func=get_rainbow_from_s,
                               linewidth=3, linestyle="dashdot", marker="^", markersize=2, alpha=0.5)
    colors_list = ["grey", "grey", "grey", "grey"]
    lines_list = [Line2D([0], [0], color="grey", linewidth=0, linestyle="solid", marker="o", markersize=10),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="dashed", marker="^", markersize=2),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="dotted", marker="^", markersize=2),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="dashdot", marker="^", markersize=2),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="s", markersize=2)]

    labels_list = ["data", "Grouplasso", "MTW", "DirtyModel", "PhysioMTL"]

    plt.legend(lines_list, labels_list, loc="upper right", fontsize=16)
    plt.title("Generalize to unseen testing tasks", fontsize=16)
    plt.xlabel("x", fontsize=18, labelpad=-5)
    plt.ylabel("f(x)", fontsize=18)
    plt.xlim(-1, 27)
    plt.ylim(30, 90)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout(pad=0)

    plt.show()
