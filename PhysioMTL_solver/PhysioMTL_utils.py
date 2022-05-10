"""
The utils functions for PhysioMTL
Data generation, visualization and so on
"""
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt


def get_sin_data(M_str, A_str, phi_str, data_num, noise, t_raw):

    noise_raw = np.random.normal(loc=0, scale=noise, size=data_num)
    y_raw = M_str + A_str * np.sin(t_raw * 0.25 + phi_str) + noise_raw

    return y_raw

# [[t_i(np), f_i(np), s_i(np)],
#  [t_i(np), f_i(np), s_i(np)],
#  ....,]
def get_rainbow_curves(s_list):

    all_data_list = []
    para_list = []

    noise_index = 0.3
    for index, s in enumerate(s_list):
        t_i_raw = np.linspace(0, 25, 30) + np.random.normal(loc=0, scale=0.0, size=30)
        noise_i_raw = np.random.normal(loc=0, scale=noise_index, size=30)
        temp_para = [35 + s*5, 3.0 + s/5, (0.34) + s * 0.000,  -1.8 + 0.4 * s]
        print("temp_para =", temp_para)
        y_i_raw = temp_para[0] + temp_para[1] * np.sin(t_i_raw * temp_para[2] + temp_para[3]) + noise_i_raw
        all_data_list.append([t_i_raw, y_i_raw, s])

        # Notice: Get the groundtruth linear parameter
        #   y_i = A * sin (r * t_i + \phi) + m
        #   y_i = theta_1 * x_i + theta_2 * z_i + theta_3
        #   theta_1 = A * cos(\phi), theta_2 = A * sin(\phi), theta_3 = m
        linear_para = [temp_para[1] * np.cos(temp_para[3]), temp_para[1] * np.sin(temp_para[3]), temp_para[0]]
        para_list.append(linear_para)

    return all_data_list, para_list


#   Notice:
#   [[t_i(np), f_i(np), s_i(np)],
#   [t_i(np), f_i(np), s_i(np)],
#   ....,]
causual_map_default = np.array([[0.2, 3.0],
                                [0.4, -1.8],
                                [5, 35]])

def underlying_truth(s):
    A = 0.2 * s + 3.0
    phi = 0.4 * s - 1.8
    M = 5 * s + 35
    return A, phi, M


def get_rainbow_curves_new(s_list_input, data_noise=0.1, underlying_func = underlying_truth):
    t_list = []     # t index for plot
    x_list = []     # X feature for regression
    s_list = []     # task feature for regression
    y_list = []     # Y label for regression

    freq = 0.34
    # Notice: generate data for each task
    for index, s in enumerate(s_list_input):
        t_i_raw = np.linspace(0, 25, 30) + np.random.normal(loc=0, scale=0.1, size=30)    # (30,)
        A_task, phi_task, M_task = underlying_func(s)
        y_i_raw = M_task + A_task * np.sin(freq * t_i_raw + phi_task) \
                  + np.random.normal(loc=0, scale=data_noise, size=30)  # (30, )

        x_i_raw = np.asarray([np.sin(freq * t_i_raw),       # (30, 3)
                                np.cos(freq * t_i_raw),
                                np.ones(30, )]).T

        t_list.append(t_i_raw)
        x_list.append(x_i_raw)
        s_list.append(s)
        y_list.append(y_i_raw)

    return t_list, x_list, s_list, y_list


# Notice: Process data for training PhysioMTL
def process_for_PhysioMTL(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
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
    ans = [n for d, n in sorted((abs(x - value), x) for x in list_input)[:k]]
    return ans


# Notice: get the model parameter mat
def k_nearest_model_para(train_W, train_s_list, test_s_list, k=2):
    w_dim, train_task_num = train_W.shape
    model_para_mat_list = []
    for s_test in test_s_list:
        result_list = k_nearest_list(s_test, train_s_list, k)
        index_list = []
        for result in result_list:
            index_list.append(train_s_list.index(result))
        weight_selected = train_W[:, index_list]    # (4, 2)
        model_para_mat_list.append(weight_selected)
    return model_para_mat_list


# Notice: get the RMSE error of two list of np
def compute_list_rmse(list_a, list_b):
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
colors_f = cm.rainbow(0.1 * np.linspace(0, 10, 101))
l_np = np.linspace(0, 10, 101)


def get_rainbow_from_s(s):
    color_select = min(list(l_np), key= lambda x:abs(x - s))
    color_index = list(l_np).index(color_select)
    return colors_f[color_index]


# Notice # scatter data samples
def scatter_data_with_s(plt, t_list_raw, Y_raw_list, S_raw_list, rainbow_func=get_rainbow_from_s, **kwargs):
    for task_i, s_value in enumerate(S_raw_list):
        plt.scatter(t_list_raw[task_i], Y_raw_list[task_i], label="task code:" + str(s_value), color=rainbow_func(s_value), **kwargs)
        # plt.plot(t_test, pred_Y_list[task_i], label="pred" + str(s_value), color=get_rainbow_from_s(s_value))
    return plt


# Notice # plot
def plot_data_curve_with_s(t_list_or_np, Y_list, S_raw_list, rainbow_func=get_rainbow_from_s, **kwargs):
    if isinstance(t_list_or_np, list):
        for task_i, s_value in enumerate(S_raw_list):
            plt.plot(t_list_or_np[task_i], Y_list[task_i], label="pred" + str(s_value), color=rainbow_func(s_value), **kwargs)





