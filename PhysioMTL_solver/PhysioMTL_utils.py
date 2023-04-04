"""
The utils functions for PhysioMTL solver and computation.
Synthetic data generation, data processing, and visualization.
"""
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np


def underlying_truth(s):
    """
    The underlying procedure that generates function values give a taskwise feature.

    Args:
        s (float): The task indicator.

    Returns:
        tuple: A tuple containing the amplitude A, phase phi, and the vertical shift M.
    """
    A = 0.2 * s + 3.0
    phi = 0.4 * s - 1.8
    M = 5 * s + 35
    return A, phi, M


def get_rainbow_curves_new(s_list_input, data_noise=0.1, underlying_func=underlying_truth):
    """
    Generate the dataset for regression.

    Args:
        s_list_input (list): A list of taskwise features.
        data_noise (float, optional): The noise for data generation. Defaults to 0.1.
        underlying_func (function, optional): The underlying sinusoidal function. Defaults to underlying_truth.

    Returns:
        tuple: A tuple containing the following four lists:
               - A list of t for each task, for plotting.
               - A list of feature X for each task, for multitask regression.
               - A list of task indicator s for each task, for multitask regression.
               - A list of target X for each task, for multitask regression.
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


def process_for_PhysioMTL(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
    """
    Reshape the data for training PhysioMTL solver.

    Args:
        raw_t_list (list): A list of t for each task, for plotting.
        raw_x_list (list): A list of feature X for each task, for multitask regression.
        raw_s_list (list): A list of taskwise features S for each task, for multitask regression.
        raw_y_list (list): A list of target X for each task, for multitask regression.

    Returns:
        tuple: A tuple containing the following four lists:
                - A list of t for each task, for plotting.
                - A list of feature X for each task, for multitask regression.
                - A list of taskwise feature S for each task, for multitask regression.
                - A list of target X for each task, for multitask regression.
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


def process_for_MTL(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
    """
    Transform the data into the format for other multitask regression methods.
    https://github.com/hichamjanati/mutar

    Args:
        raw_t_list (list): A list of t for each task, for plotting.
        raw_x_list (list): A list of feature X for each task, for multitask regression.
        raw_s_list (list): A list of taskwise features S for each task, for multitask regression.
        raw_y_list (list): A list of target X for each task, for multitask regression.

    Returns:
        tuple: A tuple containing the following two arrays:
               - An array of X_mat for multitask regression.
               - An array of Y_mat for multitask regression.
    """
    task_num = len(raw_t_list)

    X_mat = np.zeros((task_num, 30, 4))
    Y_mat = np.zeros((task_num, 30))

    for task_i, s_value in enumerate(raw_s_list):
        X_mat[task_i:task_i + 1, :, 0:3] = raw_x_list[task_i]
        X_mat[task_i:task_i + 1, :, 3] = raw_s_list[task_i]
        Y_mat[task_i:task_i + 1, :] = raw_y_list[task_i]

    return X_mat, Y_mat


def k_nearest_list(value, list_input, k):
    """
    Given a target value, return the top k nearest elements in a list.

    Args:
        value (float): A target value.
        list_input (list): A list of values.
        k (int): The number of top k elements.

    Returns:
        list: A list of the top k nearest elements.
    """
    ans = [n for d, n in sorted((abs(x - value), x) for x in list_input)[:k]]
    return ans


def k_nearest_model_para(train_W, train_s_list, test_s_list, k=2):
    """
    Find the k nearest parameters based on the taskwise indicators.

    Args:
        train_W (ndarray): The learned weight matrix from the training set with shape (feature_d, train_task_num).
        train_s_list (list): A list of task indicators for the training set.
        test_s_list (list): A list of task indicators for the testing set.
        k (int, optional): The number of top k elements. Defaults to 2.

    Returns:
        list: A list of model parameter matrices with shape (feature_d, k) for each task in the testing set.
    """
    model_para_mat_list = []
    for s_test in test_s_list:
        result_list = k_nearest_list(s_test, train_s_list, k)
        index_list = []
        for result in result_list:
            index_list.append(train_s_list.index(result))
        weight_selected = train_W[:, index_list]  # (feature_d, k)
        model_para_mat_list.append(weight_selected)
    return model_para_mat_list


def compute_list_rmse(list_a, list_b):
    """
    Compute the average RMSE error given two lists of numpy.ndarrays.

    Args:
        list_a (list): A list of ndarrays.
        list_b (list): A list of ndarrays.

    Returns:
        float: The average RMSE error.
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


def get_rainbow_from_s(s):
    """
    Get the color coefficient from taskwise indicators.

    Args:
        s (float): A taskwise indicator.

    Returns:
        ndarray: An ndarray from cm.rainbow().
    """
    colors_f = cm.rainbow(0.1 * np.linspace(0, 10, 101))
    l_np = np.linspace(0, 10, 101)
    color_select = min(list(l_np), key=lambda x: abs(x - s))
    color_index = list(l_np).index(color_select)
    return colors_f[color_index]


def scatter_data_with_s_qua(plt, t_list_raw, Y_raw_list, S_raw_list, rainbow_func=get_rainbow_from_s, **kwargs):
    """
    Scatter plots the data samples with color-coded task-wise features.

    Args:
        plt (matplotlib.pyplot object): A matplotlib.pyplot object to plot on.
        t_list_raw (list): A list of numpy arrays representing the time indices of each task.
        Y_raw_list (list): A list of numpy arrays representing the targets/labels of each task.
        S_raw_list (list): A list of numpy arrays representing the task-wise features of each task.
        rainbow_func (function, optional): A function that returns a color map based on the task-wise features.
                                            Default is get_rainbow_from_s.
        **kwargs: Additional arguments to pass to the scatter plot function.

    Returns:
        plt (matplotlib.pyplot object): A matplotlib.pyplot object with the scatter plot plotted on.
    """
    for task_i, s_value in enumerate(S_raw_list):
        plt.scatter(t_list_raw[task_i], Y_raw_list[task_i], color=rainbow_func(s_value), **kwargs)
    return plt


def plot_data_curve_with_s_qua(t_list_or_np, Y_list, S_raw_list, rainbow_func=get_rainbow_from_s, **kwargs):
    """
    Plots the data curves with color-coded task-wise features.

    Args:
        t_list_or_np (list or numpy array): A list of numpy arrays or a numpy array representing the time indices of each task.
        Y_list (list): A list of numpy arrays representing the targets/labels of each task.
        S_raw_list (list): A list of numpy arrays representing the task-wise features of each task.
        rainbow_func (function, optional): A function that returns a color map based on the task-wise features.
                                            Default is get_rainbow_from_s.
        **kwargs: Additional arguments to pass to the plot function.

    Returns:
        plt (matplotlib.pyplot object): A matplotlib.pyplot object with the scatter plot plotted on.
    """
    if isinstance(t_list_or_np, list):
        for task_i, s_value in enumerate(S_raw_list):
            plt.plot(t_list_or_np[task_i], Y_list[task_i], color=rainbow_func(s_value), **kwargs)