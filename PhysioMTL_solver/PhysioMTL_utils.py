"""
The utils functions for PhysioMTL solver and computation.
Synthetic data generation, data processing, and visualization.
"""
import matplotlib.cm as cm
import numpy as np


def underlying_truth(s):
    """ The underlying procedure that generates function values
    give a taskwise feature.
    :param s: float, the task indicator
    :return: The amplitude A, phase phi, and the vertical shift M.
    """
    A = 0.2 * s + 3.0
    phi = 0.4 * s - 1.8
    M = 5 * s + 35
    return A, phi, M


def get_rainbow_curves_new(s_list_input, data_noise=0.1, underlying_func=underlying_truth):
    """
    Generate the dataset for regression
    :param s_list_input: list of , A list of taskwise features
    :param data_noise: float, defaults to 0.1. The noise for data generation.
    :param underlying_func: python function. The underlying sinusoidal function.
    :return: A list of t for each task, for plotting.
             A list of feature X for each task, for multitask regression.
             A list of task indicator s for each task, for multitask regression.
             A list of target X for each task, for multitask regression.
    """
    t_list = []  # t index for plot
    x_list = []  # X feature for regression
    s_list = []  # task feature for regression
    y_list = []  # Y label for regression

    freq = 0.34
    # Notice: generate data for each task
    for index, s in enumerate(s_list_input):
        t_i_raw = np.linspace(0, 25, 30) + np.random.normal(loc=0, scale=0.1, size=30)  # (30,)
        A_task, phi_task, M_task = underlying_func(s)
        y_i_raw = M_task + A_task * np.sin(freq * t_i_raw + phi_task) \
                  + np.random.normal(loc=0, scale=data_noise, size=30)  # (30, )

        x_i_raw = np.asarray([np.sin(freq * t_i_raw),  # (30, 3)
                              np.cos(freq * t_i_raw),
                              np.ones(30, )]).T

        t_list.append(t_i_raw)
        x_list.append(x_i_raw)
        s_list.append(s)
        y_list.append(y_i_raw)

    return t_list, x_list, s_list, y_list


# Notice: Process data for training PhysioMTL
def process_for_PhysioMTL(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
    """
    Reshape the data for PhysioMTL solver. Reparamterize the taskwise indicators as
    taskwise features.
    :return: A list of t for each task, for plotting.
             A list of feature X for each task, for multitask regression.
             A list of taskwise feature S for each task, for multitask regression.
             A list of target X for each task, for multitask regression.
    """
    X_train_list = []
    S_train_list = []
    Y_train_list = []

    for task_i, s_value in enumerate(raw_s_list):
        # Notice: X_vec: (30, 3) -> (30, 3)
        X_train_list.append(raw_x_list[task_i])
        # Notice: Y_vec: (30, ) -> (30, 1)
        Y_train_list.append(raw_y_list[task_i].reshape((-1, 1)))
        # Notice:! S:    number -> [s, sin(s), cos(s), 1].T  (4, 1)
        S_vec = np.array([[s_value],
                          [np.sin(s_value)],
                          [np.cos(s_value)],
                          [1]])
        S_train_list.append(S_vec)

    return raw_t_list, X_train_list, S_train_list, Y_train_list


# Notice: process data for training other MTL
#       https://github.com/hichamjanati/mutar
def process_for_MTL(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
    """
    Transform the PhysioMTL data into the format for other multitask regressions methods.
    Combine the taskwise feature into the input feature X.
    :return: X_mat, Y_mat
    """
    task_num = len(raw_t_list)

    X_mat = np.zeros((task_num, 30, 4))
    Y_mat = np.zeros((task_num, 30))

    for task_i, s_value in enumerate(raw_s_list):
        X_mat[task_i:task_i + 1, :, 0:3] = raw_x_list[task_i]
        X_mat[task_i:task_i + 1, :, 3] = raw_s_list[task_i]
        Y_mat[task_i:task_i + 1, :] = raw_y_list[task_i]

    return X_mat, Y_mat


# Notice: k_nearest for normal MTL generalization
def k_nearest_list(value, list_input, k):
    """
    Given a target value, return top k nearest elements in a list.
    :param value: float, a target value.
    :param list_input: list, a list of values.
    :param k: The number of top k elements.
    :return: list of values.
    """
    ans = [n for d, n in sorted((abs(x - value), x) for x in list_input)[:k]]
    return ans


# Notice: get the model parameter mat
def k_nearest_model_para(train_W, train_s_list, test_s_list, k=2):
    """
    Use the taskwise indicators to find the k nearest parameters.
    :param train_W: The learned weight matrix from the training set (feature_d, train_task_num)
    :param train_s_list: The list of task indicators for the training set.
    :param test_s_list:The list of task indicators for the testing set.
    :param k: The number of top k elements.
    :return: (feature_d, test_task_num)
    """
    model_para_mat_list = []
    for s_test in test_s_list:
        result_list = k_nearest_list(s_test, train_s_list, k)
        index_list = []
        for result in result_list:
            index_list.append(train_s_list.index(result))
        weight_selected = train_W[:, index_list]  # (4, 2)
        model_para_mat_list.append(weight_selected)
    return model_para_mat_list


# Notice: get the RMSE error of two list of np
def compute_list_rmse(list_a, list_b):
    """
    Compute the average RMSE error given to list of numpy.ndarrays.
    :param list_a: List of ndarrays
    :param list_b: List of ndarrays
    :return: float
    """
    if len(list_a) != len(list_b):
        print("Something is wrong")
    num_task = len(list_a)
    rmse = 0
    for l_i in range(num_task):
        array_a = list_a[l_i]
        array_b = list_b[l_i]
        temp_rmse = np.sqrt(np.mean(np.square(array_a - array_b)))
        rmse += temp_rmse
    return rmse / num_task


# Notice: Visualize the task relation by rainbow color
def get_rainbow_from_s(s):
    """
    Get the color coefficient from taskwise indicators.
    :param s: float
    :return: numpy.ndarray from cm.rainbow()
    """
    colors_f = cm.rainbow(0.1 * np.linspace(0, 10, 101))
    l_np = np.linspace(0, 10, 101)
    color_select = min(list(l_np), key=lambda x: abs(x - s))
    color_index = list(l_np).index(color_select)
    return colors_f[color_index]
