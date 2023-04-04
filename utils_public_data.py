"""
The utils functions for PhysioMTL solver and computation on the public dataset.
Data processing, and visualization.
"""
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

human_feq = 2.0 * np.pi / 24    # The frequency factor for human diurnal cycle.


def compute_list_rmse(list_a, list_b):
    """
    Computes the root mean squared error (RMSE) between two lists of numpy arrays.

    Args:
        list_a (list): A list of numpy arrays.
        list_b (list): A list of numpy arrays with the same length as list_a.

    Returns:
        float: The RMSE between the two lists of numpy arrays.

    Raises:
        ValueError: If the two input lists have different lengths.

    """
    if len(list_a) != len(list_b):
        print("Something is wrong")
    num_task = len(list_a)
    rmse = 0
    for l_i in range(num_task):
        array_a = list_a[l_i]
        array_b = list_b[l_i]
        rmse += np.sqrt(np.mean(np.square(array_a - array_b)))
    return rmse / num_task


def get_baseline_MTL_mse(baseline_model_obj, X_train_mat, Y_train_mat, raw_train_tuple, raw_test_tuple, cost_function):
    """
    Trains a baseline multitask regression model and evaluates the root mean squared error (RMSE) on the test set.

    Args:
        baseline_model_obj (object): Python object representing the multitask learning model.
        X_train_mat (numpy.ndarray): Training feature matrix obtained from `process_for_MTL_pubdata()`.
        Y_train_mat (numpy.ndarray): Training target/label matrix obtained from `process_for_MTL_pubdata()`.
        raw_train_tuple (tuple): Tuple of training feature, target, and task-wise feature matrices obtained from `divide_raw_train_test_list()`.
        raw_test_tuple (tuple): Tuple of test feature, target, and task-wise feature matrices obtained from `divide_raw_train_test_list()`.
        cost_function (function): The cost metric used to compute the similarity between tasks.

    Returns:
        float: The average RMSE on the test set.

    """
    baseline_model_obj.fit(X_train_mat, Y_train_mat)
    prior_mean = np.median(Y_train_mat)
    model_para_list = k_nearest_model_para_pub(baseline_model_obj.coef_,
                                               raw_train_tuple[2], raw_test_tuple[2],
                                               cost_function)
    Y_pred_list = get_pred_Y_test_mtl(model_para_list, raw_test_tuple[1], raw_test_tuple[2], prior_mean)
    mse = compute_list_rmse(Y_pred_list, raw_test_tuple[3])
    return mse


def get_pred_Y_test_mtl(model_para_list, x_raw_test_list, s_raw_test_list, prior_mean=0):
    """
    Predicts target values given test features and model parameters.

    Args:
        model_para_list (list of numpy arrays): A list of numpy arrays representing the model parameters for each task.
        x_raw_test_list (list of numpy arrays): A list of numpy arrays representing the test features for each task.
        s_raw_test_list (list of numpy arrays): A list of numpy arrays representing the task-wise features for each task.
        prior_mean (float, optional): A scalar value representing the prior mean. Defaults to 0.

    Returns:
        list of numpy arrays: A list of numpy arrays representing the predicted target values for each task.
        
    """
    Y_pred_list = []

    for task_i, X_np in enumerate(x_raw_test_list):
        sample_num = X_np.shape[0]
        X_mat = np.zeros((sample_num, 3 + 6))
        X_mat[:, 0:3] = X_np
        X_mat[:, 3:] = s_raw_test_list[task_i].reshape((1, -1))
        Y_pred = X_mat @ model_para_list[task_i] + prior_mean
        Y_pred_list.append(Y_pred)

    return Y_pred_list


def k_nearest_model_para_pub(train_W, train_s_list, test_s_list, my_cost_function, k=1):
    """
    Finds the k nearest parameters using the task-wise indicators on a public dataset.

    Args:
        train_W (numpy.ndarray): Matrix of shape (d_feature, t), representing the learned feature weights of the model.
        train_s_list (list): List of t numpy ndarrays of shape (d_task-wise-feature, d_label), representing the task-wise feature vectors for the training set.
        test_s_list (list): List of t numpy ndarrays of shape (d_task-wise-feature, d_label), representing the task-wise feature vectors for the test set.
        my_cost_function (function): The cost function used to compute the similarity between task-wise features.
        k (int): The number of nearest neighbors to select. Defaults to 1.

    Returns:
        list: List of t numpy ndarrays of shape (d_feature, k), representing the selected weight matrices for each task in the test set.

    """
    model_para_mat_list = []
    for s_test in test_s_list:
        result_list = k_nearest_list_pub(s_test, train_s_list, k, my_cost_function)
        index_list = []
        for result in result_list:
            for idx, train_s_vec in enumerate(train_s_list):
                if np.max(train_s_vec - result) < 1e-6:
                    index_list.append(idx)
                    break
        weight_selected = train_W[:, index_list]  # (d_feature, k)
        model_para_mat_list.append(weight_selected)
    return model_para_mat_list


def k_nearest_list_pub(value, list_input, k, my_cost_function):
    """
    Finds the top k nearest elements in a list to a target value, according to a user-specified cost function.

    Args:
        value (numpy.ndarray): The target value to compare against.
        list_input (list): The list of elements to compare against the target value.
        k (int): The number of nearest neighbors to select.
        my_cost_function (function): The cost function used to compute the similarity between the elements.

    Returns:
        list: The top k nearest elements in the list to the target value.

    """
    ans = [n for d, n in sorted((my_cost_function(x, value), x) for x in list_input)[:k]]
    return ans


def process_for_MTL_pubdata(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
    """
    Transforms raw data into the format required by other multitask regression methods.

    Args:
        raw_t_list (list): List of task IDs.
        raw_x_list (list): List of numpy ndarrays, each representing the feature matrix for a given task.
        raw_s_list (list): List of numpy ndarrays, each representing the task-wise feature vectors for a given task.
        raw_y_list (list): List of numpy ndarrays, each representing the target/label variable for a given task.

    Returns:
        tuple: A tuple containing:
            - X_mat (numpy.ndarray): A tensor of shape (t, 100, d_feature), representing the combined feature matrix for all tasks, where t is the number of tasks and d_feature is the total number of features after task-wise features have been combined.
            - Y_mat (numpy.ndarray): A matrix of shape (t, 100), representing the target/label variable matrix for all tasks, where t is the number of tasks.

    """
    task_num = len(raw_t_list)
    X_mat = np.zeros((task_num, 100, 3 + 6))
    Y_mat = np.zeros((task_num, 100))

    for task_i, s_vec in enumerate(raw_s_list):
        X_raw_np = raw_x_list[task_i]
        Y_raw_np = raw_y_list[task_i]
        sample_num = X_raw_np.shape[0]
        index_subject = list(range(0, sample_num))
        random.shuffle(index_subject)

        X_selected = X_raw_np[index_subject[:100]]
        Y_selected = Y_raw_np[index_subject[:100]]
        X_mat[task_i:task_i + 1, :, 0:3] = X_selected
        Y_mat[task_i:task_i + 1, :] = Y_selected
        X_mat[task_i:task_i + 1, :, 3:] = s_vec.reshape((1, -1))

    return X_mat, Y_mat


def divide_raw_train_test_list(t_raw_list_input, X_raw_list_input, S_raw_list_input, Y_raw_list_input, tr_ratio=0.8):
    """
    Randomly divides raw data into training and test sets.

    Args:
        t_raw_list_input (list): List of numpy ndarrays representing the time indices for each task.
        X_raw_list_input (list): List of numpy ndarrays, each representing the feature matrix for a given task.
        S_raw_list_input (list): List of numpy ndarrays, each representing the task-wise feature vectors for a given task.
        Y_raw_list_input (list): List of numpy ndarrays, each representing the target/label variable for a given task.
        tr_ratio (float, optional): The ratio of training to test split. Defaults to 0.8.

    Returns:
        tuple: A tuple containing two tuples, each of which contains four numpy ndarrays representing the time indices, feature matrices, task-wise feature vectors, and target/label variables for the training and test sets.

    """
    data_num = len(t_raw_list_input)
    index_list = list(range(0, data_num))
    random.shuffle(index_list)
    train_index = index_list[:int(tr_ratio * data_num)]
    test_index = index_list[int(tr_ratio * data_num):]

    t_list_train = [t_raw_list_input[i] for i in train_index]
    X_list_train = [X_raw_list_input[i] for i in train_index]
    S_list_train = [S_raw_list_input[i] for i in train_index]
    Y_list_train = [Y_raw_list_input[i] for i in train_index]

    t_list_test = [t_raw_list_input[i] for i in test_index]
    X_list_test = [X_raw_list_input[i] for i in test_index]
    S_list_test = [S_raw_list_input[i] for i in test_index]
    Y_list_test = [Y_raw_list_input[i] for i in test_index]

    tXSY_train = (t_list_train, X_list_train, S_list_train, Y_list_train)
    tXSY_test = (t_list_test, X_list_test, S_list_test, Y_list_test)

    return tXSY_train, tXSY_test


def process_for_PhysioMTL_pubdata(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
    """
    Transforms the raw data into the format for PhysioMTL methods.

    Combines the task-wise features into the input feature X for experiments run on the public real-world dataset.

    Args:
        raw_t_list (list): List of numpy ndarrays representing the time indices for each task.
        raw_x_list (list): List of numpy ndarrays, each representing the feature matrix for a given task.
        raw_s_list (list): List of numpy ndarrays, each representing the task-wise feature vectors for a given task.
        raw_y_list (list): List of numpy ndarrays, each representing the target/label variable for a given task.

    Returns:
        tuple: A tuple containing four numpy ndarrays representing the time indices, feature matrices, task-wise feature vectors, and target/label variables for each task in the format required by PhysioMTL methods.

    """
    X_train_list = []
    S_train_list = []
    Y_train_list = []

    for task_i, s_vec in enumerate(raw_s_list):
        # Notice: X_vec: (30, 3) -> (30, 3)
        X_train_list.append(raw_x_list[task_i])
        # Notice: Y_vec: (30, ) -> (30, 1)
        Y_train_list.append(raw_y_list[task_i].reshape((-1, 1)))
        # Notice:! S:    (6,) -> [s_0, s_1,...,s_5, 1].T  (7, 1)
        S_train_vec = np.hstack([s_vec, 1]).reshape(-1, 1)
        S_train_list.append(S_train_vec)

    return raw_t_list, X_train_list, S_train_list, Y_train_list


def change_labels(ax):
    """
    Changes the labels and limits of a matplotlib plot.

    Args:
        ax: The axis object to modify.

    Returns:
        None.
        
    """
    ax.legend(loc="lower right", fontsize=14)

    # Notice: 61, 89 for kernel map
    # Notice: (65, 120) for linear map
    ax.set_ylim(65, 100)

    ax.set_xticks([9, 12, 15, 18, 21, 24, 27, 30, 33, 36])
    ax.set_xticklabels(["9:00", "12:00", "15:00", "18:00", "21:00", "24:00",
                        "3:00", "6:00", "9:00", "12:00"], fontsize=16, rotation=30)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_ylabel("HRV", fontsize=18)
    ax.set_xlabel("Time", fontsize=18)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')


def investigate_all_model_save(model, s_vec_base):
    """
    Do the counterfactual analysis, plot and save the results.
    
    Args:
        model: The PhysioMTL solver with nonlinear map.
        s_vec_base: The baseline taskwise demographic feature.

    Returns:
        None
        
    """
    t_test = np.linspace(8, 36, 30)
    X_test = np.asarray([np.sin(human_feq * t_test),
                         np.cos(human_feq * t_test),
                         np.ones(30, )]).T

    tight_pad = 0

    kwargs_input = {"linewidth": 3}

    # Notice: Age vs HRV
    plt.figure(1)
    fig, ax = plt.subplots()
    for i in range(10):
        age_this = 20 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[0] = age_this
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0], label="age=" + str(round(age_this, 2)),
                c="black", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_kernel_2_age.png', dpi=500, bbox_inches='tight')

    # Notice: BMI vs HRV
    plt.figure(2)
    fig, ax = plt.subplots()
    for i in range(10):
        weight_this = 55 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[2] = weight_this
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0],
                label="bmi=" + str(round(weight_this / s_vec_base[1, 0] ** 2, 1)),
                c="red", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_kernel_2_bmi.png', dpi=500, bbox_inches='tight')

    # Notice: Activity vs HRV
    plt.figure(3)
    fig, ax = plt.subplots()
    for i in range(10):
        activity = 0.2 + 0.3 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[3] = activity
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0],
                label="activity=" + str(round(activity, 2)),
                c="orange", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_kernel_2_activity.png', dpi=500, bbox_inches='tight')

    # Notice: Sleep vs HRV
    plt.figure(4)
    fig, ax = plt.subplots()
    for i in range(10):
        sleep_this = 3 + 0.8 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[4] = sleep_this
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0],
                label="sleep=" + str(round(sleep_this, 2)),
                c="purple", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_kernel_2_sleep.png', dpi=500, bbox_inches='tight')

    # Notice: Stress vs HRV
    plt.figure(5)
    fig, ax = plt.subplots()
    for i in range(10):
        stress_this = 20 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[5] = stress_this
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0],
                label="stress =" + str(stress_this),
                c="green", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_kernel_2_stress.png', dpi=500, bbox_inches='tight')

    # Notice: Get the baseline
    plt.figure(6)
    fig, ax = plt.subplots()

    s_vec_this = s_vec_base.copy()
    y_pred = model.predict([X_test], [s_vec_this])
    ax.plot(t_test, y_pred[0],
            c="blue", alpha=1, linestyle="--", linewidth=5)
    lines_list = [Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0)]

    labels_list = ["Age=40 (years)", "BMI=25 (kg/m^2)", "Sleep=7 (hours)",
                   "Activity=1.5 (hours)", "Stress=40 (unit)"]

    ax.set_ylim(65, 100)
    ax.set_xticks([9, 12, 15, 18, 21, 24, 27, 30, 33, 36])
    ax.set_xticklabels(["9:00", "12:00", "15:00", "18:00", "21:00", "24:00",
                        "3:00", "6:00", "9:00", "12:00"], fontsize=16, rotation=30)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_ylabel("HRV", fontsize=18)
    ax.set_xlabel("Time", fontsize=18)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.legend(lines_list, labels_list, loc="lower right", fontsize=14)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_kernel_2_baseline.png', dpi=500, bbox_inches='tight')


def investigate_all_model_save_linear(model, s_vec_base):
    """
    Do the counterfactual analysis, plot and save the results.
    
    Args:
        model: The PhysioMTL solver with nonlinear map.
        s_vec_base: The baseline taskwise demographic feature.

    Returns:
        None
        
    """
    t_test = np.linspace(8, 36, 30)
    X_test = np.asarray([np.sin(human_feq * t_test),
                         np.cos(human_feq * t_test),
                         np.ones(30, )]).T

    tight_pad = 0

    kwargs_input = {"linewidth": 3}

    # Notice: Age vs HRV
    plt.figure(1)
    fig, ax = plt.subplots()
    for i in range(10):
        age_this = 20 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[0] = age_this
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0], label="age=" + str(round(age_this, 2)),
                c="black", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_linear_2_age.png', dpi=500, bbox_inches='tight')

    # Notice: BMI vs HRV
    plt.figure(2)
    fig, ax = plt.subplots()
    for i in range(10):
        weight_this = 55 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[2] = weight_this
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0],
                label="bmi=" + str(round(weight_this / s_vec_base[1, 0] ** 2, 1)),
                c="red", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_linear_2_bmi.png', dpi=500, bbox_inches='tight')

    # Notice: Activity vs HRV
    plt.figure(3)
    fig, ax = plt.subplots()
    for i in range(10):
        activity = 0.2 + 0.3 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[3] = activity
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0],
                label="activity=" + str(round(activity, 2)),
                c="orange", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_linear_2_activity.png', dpi=500, bbox_inches='tight')

    # Notice: Sleep vs HRV
    plt.figure(4)
    fig, ax = plt.subplots()
    for i in range(10):
        sleep_this = 3 + 0.8 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[4] = sleep_this
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0],
                label="sleep=" + str(round(sleep_this, 2)),
                c="purple", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_linear_2_sleep.png', dpi=500, bbox_inches='tight')

    # Notice: Stress vs HRV
    plt.figure(5)
    fig, ax = plt.subplots()
    for i in range(10):
        stress_this = 20 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[5] = stress_this
        y_pred = model.predict([X_test], [s_vec_this])
        ax.plot(t_test, y_pred[0],
                label="stress =" + str(stress_this),
                c="green", alpha=0.1 + 0.09 * i, **kwargs_input)
    change_labels(ax)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_linear_2_stress.png', dpi=500, bbox_inches='tight')

    # Notice: Get the baseline
    plt.figure(6)
    fig, ax = plt.subplots()

    s_vec_this = s_vec_base.copy()
    y_pred = model.predict([X_test], [s_vec_this])
    ax.plot(t_test, y_pred[0],
            c="blue", alpha=1, linestyle="--", linewidth=5)
    lines_list = [Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0),
                  Line2D([0], [0], color="grey", linewidth=3, linestyle="solid", marker="^", markersize=0)]

    labels_list = ["Age=40 (years)", "BMI=25 (kg/m^2)", "Sleep=7 (hours)",
                   "Activity=1.5 (hours)", "Stress=40 (unit)"]

    # ax.set_ylim(61, 89) Notice: (61, 89) for kernel method
    ax.set_ylim(65, 120)
    ax.set_xticks([9, 12, 15, 18, 21, 24, 27, 30, 33, 36])
    ax.set_xticklabels(["9:00", "12:00", "15:00", "18:00", "21:00", "24:00",
                        "3:00", "6:00", "9:00", "12:00"], fontsize=16, rotation=30)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_ylabel("HRV", fontsize=18)
    ax.set_xlabel("Time", fontsize=18)

    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.legend(lines_list, labels_list, loc="lower right", fontsize=14)
    plt.tight_layout(pad=tight_pad)
    fig.savefig('data_and_pickle/figures/MMASH_linear_2_baseline.png', dpi=500, bbox_inches='tight')


def get_raw_list_from_public_data_custom(data_dict_input, removed_list):
    """
    Converts the preprocessed MMASH data for multitask regression models.

    Args:
        data_dict_input (dict): A dictionary containing the preprocessed MMASH data.
        removed_list (list, optional): A list of subject IDs to be removed.

    Returns:
        tuple: A tuple containing the lists of time indices, input features, task-wise features, targets, and subject IDs.
        
    """
    human_feq = 2.0 * np.pi / 24
    
    key_list = list(data_dict_input.keys())
    t_raw_list = []
    X_raw_list = []
    S_raw_list = []
    Y_raw_list = []
    subject_id_list = []
    for key in key_list:
        if key in removed_list:
            continue
        t_np, y_np, s_vec = data_dict_input[key]
        sample_num = t_np.shape[0]
        x_raw = np.asarray([np.sin(human_feq * t_np),
                            np.cos(human_feq * t_np),
                            np.ones(sample_num, )]).T

        # Naive imputation methods
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