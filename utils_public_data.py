import numpy as np
from matplotlib import pyplot as plt
import random
from matplotlib.lines import Line2D

human_feq = 2.0 * np.pi / 24


# Notice: get the RMSE error of two list of np
def compute_list_rmse(list_a, list_b):
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

    baseline_model_obj.fit(X_train_mat, Y_train_mat)
    prior_mean = np.median(Y_train_mat)
    model_para_list = k_nearest_model_para_pub(baseline_model_obj.coef_,
                                               raw_train_tuple[2], raw_test_tuple[2],
                                               cost_function)
    Y_pred_list = get_pred_Y_test_mtl(model_para_list, raw_test_tuple[1], raw_test_tuple[2], prior_mean)
    mse = compute_list_rmse(Y_pred_list, raw_test_tuple[3])
    return mse


def get_pred_Y_test_mtl(model_para_list, x_raw_test_list, s_raw_test_list, prior_mean=0):
    
    Y_pred_list = []
    
    for task_i, X_np in enumerate(x_raw_test_list):
        sample_num = X_np.shape[0]
        X_mat = np.zeros((sample_num, 3 + 6))
        X_mat[:, 0:3] = X_np
        X_mat[:, 3:] = s_raw_test_list[task_i].reshape((1, -1))
        Y_pred = X_mat @ model_para_list[task_i] + prior_mean
        Y_pred_list.append(Y_pred)

    return Y_pred_list


def k_nearest_list_pub(value, list_input, k, my_cost_function):
    ans = [n for d, n in sorted((my_cost_function(x, value), x) for x in list_input)[:k]]
    return ans


# Notice: get the model parameter mat
def k_nearest_model_para_pub(train_W, train_s_list, test_s_list, my_cost_function, k=1):
    w_dim, train_task_num = train_W.shape
    model_para_mat_list = []
    for s_test in test_s_list:
        result_list = k_nearest_list_pub(s_test, train_s_list, k, my_cost_function)
        index_list = []
        for result in result_list:
            for idx, train_s_vec in enumerate(train_s_list):
                if np.max(train_s_vec - result) < 1e-6:
                    index_list.append(idx)
                    break
        weight_selected = train_W[:, index_list]    # (4, 2)
        model_para_mat_list.append(weight_selected)
    return model_para_mat_list


def process_for_MTL_pubdata(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
    
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
        X_mat[task_i:task_i+1, :, 0:3] = X_selected
        Y_mat[task_i:task_i+1,:] = Y_selected
        X_mat[task_i:task_i+1, :, 3:] = s_vec.reshape((1, -1))

    return X_mat, Y_mat


def divide_raw_train_test_list(t_raw_list_input, X_raw_list_input, S_raw_list_input, Y_raw_list_input, tr_ratio=0.8):
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


def get_raw_list_from_public_data(data_dict_input):
    key_list = list(data_dict_input.keys())
    t_raw_list = []
    X_raw_list = []
    S_raw_list = []
    Y_raw_list = []
    for key in key_list:
        t_np, y_np, s_vec = data_dict_input[key]
        sample_num = t_np.shape[0]
        s_vec[1] = s_vec[1] / 100

        x_raw = np.asarray([np.sin(human_feq * t_np),
                            np.cos(human_feq * t_np),
                            np.ones(sample_num, )]).T
        t_raw_list.append(t_np)
        X_raw_list.append(x_raw)
        S_raw_list.append(s_vec)
        Y_raw_list.append(y_np)
        # break
    return t_raw_list, X_raw_list, S_raw_list, Y_raw_list


# Notice: just
def process_for_PhysioMTL_pubdata(raw_t_list, raw_x_list, raw_s_list, raw_y_list):
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


# Notice functions that show the effects of different attributes
def investigate_age(T_map, s_vec_base, X_test_plot, t_test_plot):
    for i in range(10):
        age_this = 20 + 1.5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[0] = age_this
        W_this = T_map @ s_vec_this
        y_pred = X_test_plot @ W_this
        plt.plot(t_test_plot, y_pred, label="age=" + str(round(age_this, 3)), c="black", alpha=0.1 + 0.09 * i)
    plt.legend()


def investigate_weight(T_map, s_vec_base, X_test_plot, t_test_plot):
    for i in range(10):
        weight_this = 55 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[2] = weight_this
        W_this = T_map @ s_vec_this
        y_pred = X_test_plot @ W_this
        plt.plot(t_test_plot, y_pred,
                 label="bmi=" + str(round(weight_this / s_vec_base[1, 0] ** 2, 3)),
                 c="red", alpha=0.1 + 0.09 * i)
    plt.legend()


def investigate_activity(T_map, s_vec_base, X_test_plot, t_test_plot):
    for i in range(10):
        activity = 0.2 + 0.3 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[3] = activity
        W_this = T_map @ s_vec_this
        y_pred = X_test_plot @ W_this
        plt.plot(t_test_plot, y_pred,
                 label="activity=" + str(round(activity, 3)),
                 c="orange", alpha=0.1 + 0.09 * i)
    plt.legend()


def investigate_sleep(T_map, s_vec_base, X_test_plot, t_test_plot):
    for i in range(10):
        sleep_this = 3 + 0.8 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[4] = sleep_this
        W_this = T_map @ s_vec_this
        y_pred = X_test_plot @ W_this
        plt.plot(t_test_plot, y_pred,
                 label="sleep=" + str(round(sleep_this,2)),
                 c="purple", alpha=0.1 + 0.09 * i)
    plt.legend()


def investigate_stress(T_map, s_vec_base, X_test_plot, t_test_plot):
    for i in range(10):
        stress_this = 20 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[5] = stress_this
        W_this = T_map @ s_vec_this
        y_pred = X_test_plot @ W_this
        plt.plot(t_test_plot, y_pred,
                 label="stress =" + str(stress_this),
                 c="green", alpha=0.1 + 0.09 * i)
    plt.legend()


# Notice: overall...
def investigate_all(T_map, s_vec_base):
    t_test = np.linspace(8, 36, 30)
    X_test = np.asarray([np.sin(human_feq * t_test),
                         np.cos(human_feq * t_test),
                         np.ones(30, )]).T

    plt.figure(1)
    investigate_age(T_map=T_map, s_vec_base=s_vec_base, X_test_plot=X_test, t_test_plot=t_test)
    plt.show()

    plt.figure(2)
    investigate_weight(T_map=T_map, s_vec_base=s_vec_base, X_test_plot=X_test, t_test_plot=t_test)
    plt.show()

    plt.figure(3)
    investigate_activity(T_map=T_map, s_vec_base=s_vec_base, X_test_plot=X_test, t_test_plot=t_test)
    plt.show()

    plt.figure(4)
    investigate_sleep(T_map=T_map, s_vec_base=s_vec_base, X_test_plot=X_test, t_test_plot=t_test)
    plt.show()

    plt.figure(5)
    investigate_stress(T_map=T_map, s_vec_base=s_vec_base, X_test_plot=X_test, t_test_plot=t_test)
    plt.show()


# Notice:
def investigate_all_model(model, s_vec_base):
    t_test = np.linspace(8, 36, 30)
    X_test = np.asarray([np.sin(human_feq * t_test),
                         np.cos(human_feq * t_test),
                         np.ones(30, )]).T

    # Notice: Age vs HRV
    plt.figure(1)
    for i in range(10):
        age_this = 20 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[0] = age_this
        y_pred = model.predict([X_test], [s_vec_this])
        plt.plot(t_test, y_pred[0], label="age=" + str(round(age_this, 3)), c="black", alpha=0.1 + 0.09 * i)
    plt.legend()

    # Notice: BMI vs HRV
    plt.figure(2)
    for i in range(10):
        weight_this = 55 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[2] = weight_this
        y_pred = model.predict([X_test], [s_vec_this])
        plt.plot(t_test, y_pred[0],
                 label="bmi=" + str(round(weight_this / s_vec_base[1, 0] ** 2, 3)),
                 c="red", alpha=0.1 + 0.09 * i)
    plt.legend()

    # Notice: Activity vs HRV
    plt.figure(3)
    for i in range(10):
        activity = 0.2 + 0.3 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[3] = activity
        y_pred = model.predict([X_test], [s_vec_this])
        plt.plot(t_test, y_pred[0],
                 label="activity=" + str(round(activity, 2)),
                 c="orange", alpha=0.1 + 0.09 * i)
    plt.legend()

    # Notice: Sleep vs HRV
    plt.figure(4)
    for i in range(10):
        sleep_this = 3 + 0.8 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[4] = sleep_this
        y_pred = model.predict([X_test], [s_vec_this])
        plt.plot(t_test, y_pred[0],
                 label="sleep=" + str(round(sleep_this, 2)),
                 c="purple", alpha=0.1 + 0.09 * i)
    plt.legend()

    # Notice: Stress vs HRV
    plt.figure(5)
    for i in range(10):
        stress_this = 20 + 5 * i
        s_vec_this = s_vec_base.copy()
        s_vec_this[5] = stress_this
        y_pred = model.predict([X_test], [s_vec_this])
        plt.plot(t_test, y_pred[0],
                 label="stress =" + str(stress_this),
                 c="green", alpha=0.1 + 0.09 * i)
    plt.legend()


def change_labels(ax):
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
    t_test = np.linspace(8, 36, 30)
    X_test = np.asarray([np.sin(human_feq * t_test),
                         np.cos(human_feq * t_test),
                         np.ones(30, )]).T

    tight_pad = 0

    kwargs_input = {"linewidth":3}

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
    t_test = np.linspace(8, 36, 30)
    X_test = np.asarray([np.sin(human_feq * t_test),
                         np.cos(human_feq * t_test),
                         np.ones(30, )]).T

    tight_pad = 0

    kwargs_input = {"linewidth":3}

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