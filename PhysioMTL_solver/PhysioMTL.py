"""Solver class for PhysioMTL multitask regression model."""
import numpy as np


def sinkhorn_plan(cost_matrix, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Sinkhorn distance using the Sinkhorn-Knopp algorithm.

    Args:
        cost_matrix (ndarray): Cost matrix with shape (n x m).
        r (ndarray): Vector of marginals with shape (n, ).
        c (ndarray): Vector of marginals with shape (m, ).
        lam (float): Strength of the entropic regularization.
        epsilon (float, optional): Convergence parameter. Defaults to 1e-5.

    Returns:
        tuple: A tuple containing the optimal transport matrix with shape (n x m) and the Sinkhorn distance.
        
    """
    n, m = cost_matrix.shape
    P = np.exp(- lam * cost_matrix)
    P /= P.sum()  # normalize this matrix
    u = np.zeros(n)
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * cost_matrix)


class PhysioMTL:
    """
    The base PhysioMTL solver class for multitask regression.

    Args:
        alpha (float): Constant that multiplies the OT map resitmation term. Defaults to 0.1.
        aux_feat_cost_func (function/lambda function): The distance metric of the task wise features.
            It is usually set as a weighted l2 norm. It allows users to specify the similarity between tasks.
        X_data_list (list): List of t numpy.ndarrays, each with shape (n_samples, d_feature), float. Training data.
        Y_data_list (list): List of t numpy.ndarrays, each with shape (n_samples, 1), float. Training target/label variable.
        S_data_list (list): List of t numpy.ndarrays, each with shape (d_task-wise_feature, 1), float.
            The auxiliary task-wise feature vectors.
        cost_mat (numpy.ndarray, float, optional): The similarity matrix between tasks. Defaults to None.
        verbose_T_grad (bool, optional): Whether to display transport map T and regressor W values in each iteration.
            Defaults to False.
        map_type (str, optional): The type of the transport map. Must be one of {"linear", "kernel"}. Defaults to "linear".
        kernel_sigma (float, optional): The width parameter of the Gaussian RBF kernel. Defaults to 1.
        Pi (numpy.ndarray, float, optional): The optimal transport coupling. Defaults to None.
        coef_ (tuple of numpy.ndarray, float): self.W, self.T, The learned multitask parameters and the transport map.
        T_lr (float, optional): The learning rate for updating the transport map. Defaults to 1e-3.
        W_lr (float, optional): The learning rate for updating the regressors. Defaults to 1e-6.
        T_ite_num (int, optional): The iteration number for updating the transport map T. Defaults to 50.
        W_ite_num (int, optional): The iteration number for updating the regressor function weights. Defaults to 50.
        T_grad_F_norm_threshold (float, optional): The threshold that prevents gradient explosion when updating transport
            map T. Defaults to 1e-6.
        W_grad_F_norm_threshold (float, optional): The threshold that prevents gradient explosion when updating regressors
            W. Defaults to 1e-7.
        all_ite_num (int, optional): The total iteration number of the algorithm. Defaults to 50.
        
    """

    def __init__(self, alpha=0.1, T_initial=None,
                 T_lr=1e-3, W_lr=1e-6,
                 T_ite_num=50, W_ite_num=50,
                 all_ite_num=50,
                 verbose_T_grad=False,
                 map_type="linear", kernel_cost_function=None, kernel_sigma=1,
                 T_grad_F_norm_threshold=1e-6,
                 W_grad_F_norm_threshold=1e-7):
        self.alpha = alpha
        self.aux_feat_cost_func = None
        self.X_data_list = None
        self.Y_data_list = None
        self.S_data_list = None
        self.cost_mat = None
        self.verbose_T_grad = verbose_T_grad
        self.map_type = map_type
        self.kernel_sigma = kernel_sigma
        self.Pi = None
        self._train_task_n = 0
        self.T_initial = T_initial
        self.coef_ = None
        self.T_lr = T_lr
        self.W_lr = W_lr
        self.T_ite_num = T_ite_num
        self.W_ite_num = W_ite_num
        self.T_grad_F_norm_threshold = T_grad_F_norm_threshold
        self.W_grad_F_norm_threshold = W_grad_F_norm_threshold
        self.all_ite_num = all_ite_num
        if kernel_cost_function == None:
            print("Use l2 as kernel cost")

            def kernel_cost_l2(x, y):
                return np.mean(np.square(x - y))

            self.kernel_cost = kernel_cost_l2
        else:
            self.kernel_cost = kernel_cost_function
            print("kernel cost defined")

    def set_aux_feature_cost_function(self, aux_feat_cost_func):
        """
        Sets the auxiliary feature cost function used for computing the similarity between taskwise features.

        Args:
            aux_feat_cost_func (function): A Python function or lambda function that computes the similarity between taskwise features.

        Returns:
            float: The result of setting the auxiliary feature cost function.

        """
        self.aux_feat_cost_func = aux_feat_cost_func

    # Notice: Fit
    def fit(self, X_list, S_list, Y_list):
        """Fits the model using the given training data and auxiliary task-wise features.

        Args:
            X_list (list): List of t numpy ndarrays of shape (n_samples, d_feature), representing the training data.
            S_list (list): List of t numpy ndarrays of shape (d_task-wise_feature, d_label), representing the auxiliary task-wise feature vectors.
            Y_list (list): List of t numpy ndarrays of shape (n_samples, 1), representing the training target/label variable.

        Returns:
            None: This method does not return any values.

        """
        self.X_data_list = X_list
        self.S_data_list = S_list
        self.Y_data_list = Y_list

        n_task = len(X_list)
        self._train_task_n = n_task

        if self.map_type not in ["linear", "kernel"]:
            print("Wrong, set map='linear' or 'kernel'!")
            return 0

        # Notice: Step 1. Initialize w with independent regression
        W_list = []

        for i in range(n_task):
            X_t = X_list[i].T  # (3, 30)
            Y_t = Y_list[i].T  # (1, 30)
            W_est = Y_t @ X_t.T @ np.linalg.inv(X_t @ (X_t.T))  # (1, 3)
            W_list.append(W_est)
        W_np = np.concatenate(W_list).T  # (w_dim=3, n_task)

        # Notice: Step 2. Get the cost matrix from domain knowledge
        self.cost_mat = np.zeros((n_task, n_task))
        for i in range(n_task):
            for j in range(n_task):
                self.cost_mat[i, j] = self.aux_feat_cost_func(self.S_data_list[i],
                                                              self.S_data_list[j])

        # Notice: Step 2.5, Get K_S instead of S if using kernel map
        if self.map_type == "linear":
            S_np = np.concatenate(S_list, axis=1)  # (s_dim, n_task)
        elif self.map_type == "kernel":
            # First get a cost matrix (data_num, data_num)
            #  where each element is c(s_i, s_j)
            #  then get K_ij by exp(- c / r**2)
            temp_cost_mat = np.zeros((n_task, n_task))
            for i in range(n_task):
                for j in range(n_task):
                    temp_cost_mat[i, j] = self.kernel_cost(self.S_data_list[i],
                                                           self.S_data_list[j])
            K_mat = np.exp(- temp_cost_mat / (2 * self.kernel_sigma ** 2))
            S_np = K_mat

        w_dim, w_task_num = W_np.shape
        s_dim, s_task_num = S_np.shape  # If using kernel, S_np(K) is (data_num, data_num)
        if w_task_num != s_task_num:
            print("Wrong!!! w_task_num != s_task_num")

        # Notice: Step 3. Get the OT coupling via Sinkhorn
        v_dmy = u_dmy = np.ones(n_task) / n_task
        self.Pi, w_d = sinkhorn_plan(cost_matrix=self.cost_mat, r=v_dmy, c=u_dmy,
                                     lam=5, epsilon=1e-4)

        if self.T_initial is None:
            self.T_initial = np.zeros((w_dim, s_dim))
        T_gd = self.T_initial

        # Notice: Step 4: Iteratively update T and W
        for i_ite in range(self.all_ite_num):

            # Notice: Fix W update T
            T_grad_F_norm_last = 1e6
            for t_ite in range(self.T_ite_num):
                grad_T = np.zeros((w_dim, s_dim))

                for i in range(n_task):
                    for j in range(n_task):
                        grad_T = grad_T - self.alpha * 2 * self.Pi[i, j] * \
                                 (W_np[:, i:i + 1] - T_gd @ S_np[:, j:j + 1]) @ \
                                 S_np[:, j:j + 1].T

                T_gd = T_gd - self.T_lr * grad_T

                # Notice: Check convergence
                T_grad_F_norm = np.sum(np.square(grad_T))
                T_grad_F_norm_diff = np.abs(T_grad_F_norm_last - T_grad_F_norm)
                T_grad_F_norm_last = T_grad_F_norm
                if T_grad_F_norm_diff < self.T_grad_F_norm_threshold:
                    print("Early terminate for T at t_ite =", t_ite)
                    break

            # Notice: Fix T, update W
            W_grad_F_norm_last = 1e6
            for w_ite in range(self.W_ite_num):
                grad_W = np.zeros_like(W_np)  # (3, 5)

                for t_ite in range(n_task):  # Notice: Compute the w_grad oen by one

                    w_t_grad = (W_np[:, t_ite:t_ite + 1].T @ X_list[t_ite].T - Y_list[t_ite].T) \
                               @ (X_list[t_ite])

                    # Notice: gradient with regard
                    for t_t_ite in range(n_task):
                        w_t_grad = w_t_grad - self.alpha * (
                                T_gd @ S_np[:, t_t_ite:t_t_ite + 1] - W_np[:, t_t_ite:t_t_ite + 1]).T
                    grad_W[:, t_ite:t_ite + 1] = w_t_grad.T

                W_np = W_np - self.W_lr * grad_W

                # Notice: Check convergence
                W_grad_F_norm = np.sum(np.square(grad_W))
                W_grad_F_norm_diff = np.abs(W_grad_F_norm_last - W_grad_F_norm)
                W_grad_F_norm_last = W_grad_F_norm
                if W_grad_F_norm_diff < self.W_grad_F_norm_threshold:
                    print("Early terminate for W at w_ite =", w_ite)
                    break

            if self.verbose_T_grad:
                print()
                print("i_ite =", i_ite)
                print("last grad_T =", grad_T)
                print("last T_grad_F_norm", T_grad_F_norm)

        self.W = W_np
        self.T = T_gd
        self.coef_ = self.W, self.T

    def predict(self, X_list=None, S_list=None):
        """Predicts the target/label variable using the trained model and given input data.

        Args:
            X_list (list): List of t numpy ndarrays of shape (n_samples, d_feature), representing the input data. Defaults to None.
            S_list (list): List of t numpy ndarrays of shape (d_task-wise_feature, d_label), representing the auxiliary task-wise feature vectors. Defaults to None.

        Returns:
            list: List of t numpy ndarrays of shape (n_samples, 1), representing the predicted target/label variable.

        """
        if (X_list is None) and (S_list is None):
            pred_Y_list = []
            for i, X_np in enumerate(self.X_data_list):
                W_task = self.W[:, i:i + 1]  # (3,1)
                pred_Y = self.X_data_list[i] @ W_task
                pred_Y_list.append(pred_Y)
        else:
            # predict for unknown tasks
            pred_Y_list = []
            test_task_n = len(S_list)
            S_list_test = []
            if self.map_type == "linear":
                S_list_test = S_list
            elif self.map_type == "kernel":
                temp_cost_mat = np.zeros((self._train_task_n, test_task_n))
                for i in range(self._train_task_n):
                    for j in range(test_task_n):
                        temp_cost_mat[i, j] = self.kernel_cost(self.S_data_list[i],
                                                               S_list[j])
                K_mat = np.exp(- temp_cost_mat / (2 * self.kernel_sigma ** 2))
                for j in range(test_task_n):
                    S_list_test.append(K_mat[:, j:j + 1])

            for i, X_np in enumerate(X_list):
                W_pred = self.T @ S_list_test[i]
                pred_test_Y = X_np @ W_pred
                pred_Y_list.append(pred_test_Y)

        return pred_Y_list

