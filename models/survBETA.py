import numpy as np
import torch
import cvxpy as cp
from models.beran import Beran
from pandas.core.common import flatten
from sklearn.neighbors import NearestNeighbors

class Ensemble_Beran():
    def __init__(self, omega = 1, tau = 1, maximum_number_of_pairs = 10,
                 n_estimators = 10, size_bagging = 0.4,
                 epsilon = 0.5, lr = 1e-1, const_in_div = 100, num_epoch = 100,
                 MAE_optimisation = True, 
                 epsilon_optimisation = True, 
                 c_index_optimisation = True,
                 mode = 'gradient') -> None:
        
        self.maximum_number_of_pairs = maximum_number_of_pairs
        self.n_estimators = n_estimators
        self.epsilon = epsilon
        self.size_bagging = size_bagging
        self.MAE_optimisation = MAE_optimisation
        self.epsilon_optimisation = epsilon_optimisation
        self.c_index_optimisation = c_index_optimisation
        self.omega = omega
        self.lr = lr
        self.const_in_div = const_in_div
        self.num_epoch = num_epoch
        self.mode = mode
        self.tau = tau
        np.random.seed(42)
        torch.manual_seed(42)

    def _ensemble_weights(self, prototype, X, omega):
        X = X[:, np.newaxis, :]   #m*1*f
        #m*k*f
        diff = prototype - X    #m*k*f
        distance = np.linalg.norm(diff, axis=2) #m*k
        weights = -(distance**2)/omega
        #m*k*t
        weights_max = np.amax(weights, axis=1, keepdims=True)
        exp_weights_shifted = np.exp(weights - weights_max)
        alpha = exp_weights_shifted/np.sum(exp_weights_shifted, axis=1, keepdims=True)  #m*k
        return alpha
    
    def _sorting(self, X_train, y_train):
        times = np.array(list(zip(*y_train))[1])
        args = np.argsort(times)
        times = times[args]
        X_train = X_train[args]
        y_train = y_train[args]
        delta = np.array(list(zip(*y_train))[0])
        return X_train, y_train, times, delta
    
    def _find_set_for_C_index_optimisation(self, y_train, n):
        left = []
        right = []
        for i in range(0, y_train.shape[0]):
            if y_train[i][0]:
                current_right = []
                for j in range(0, y_train.shape[0]):
                    if y_train[j][1]>y_train[i][1]:
                        current_right.append(j)
                        left.append(i)
                if len(current_right) > n:
                    right.append(np.random.choice(current_right, size=n, replace=False))
                    left = left[:-(len(current_right)-n)]
                elif len(current_right) <= n and len(current_right)!=0:
                    right.append(current_right)
        right = list(flatten(right))
        left = np.array(left)
        right = np.array(right)
        return left, right

    def _select_random_points_with_neighbors(self, X_train, count_clusters, count_neighbors):
        random_indices = np.random.choice(len(X_train), count_clusters, replace=False)
        knn = NearestNeighbors(n_neighbors=count_neighbors+1)
        knn.fit(X_train)
    
        clusters = []

        for idx in random_indices:
            neighbors = knn.kneighbors([self.X_train[idx]], return_distance=False)
            neighbors[0] = np.sort(neighbors[0])
            neighbor_indices = neighbors[0]
            clusters.append(neighbor_indices)
        return clusters
    
    def _train_each_beran(self, X_train, y_train, n_estimators, cluster_points, kernels, times_train, delta_train, const_in_div, C_index_optimisation, MAE_optimisation, tau):
        beran_models = []
        for iteration in range(n_estimators):
            indeces = cluster_points[iteration]
            X_train_k = X_train[indeces]
            y_train_k = y_train[indeces]
            delta_k = delta_train[indeces]
            times_k = times_train[indeces]
            left_k = []
            right_k = []
            if C_index_optimisation == True:
                left_k, right_k = self._find_set_for_C_index_optimisation(y_train_k, self.maximum_number_of_pairs)
            kernel_k = kernels[iteration]
            beran_k = Beran(kernel_k, const_in_div, tau, C_index_optimisation, MAE_optimisation)
            beran_k.train(X_train_k, times_k, delta_k, left_k, right_k)
            beran_models.append(beran_k)
        return beran_models
    
    def _find_H_S_T(self, X_train, times_train, X_test, beran_models, n_estimators, cluster_points):
        H = []
        S = []
        T = []
        delta_times = np.diff(times_train)
        for iteration in range(n_estimators):
            beran = beran_models[iteration]
            H_k = np.zeros((X_test.shape[0], X_train.shape[0]))
            indeces = cluster_points[iteration]
            H_k[:, indeces] = beran.predict_cumulative_risk(X_test)
            S_k = np.exp(-np.cumsum(H_k, axis=1))
            S_k_without_last = S_k[:, :-1]
            T_k = np.einsum('mt,t->m', S_k_without_last, delta_times)
            H.append(H_k)
            S.append(S_k)
            T.append(T_k)
        return np.stack(H, axis=1), np.stack(S, axis=1), np.stack(T, axis=1)
    
    def _find_prototypes(self, X_test, n_estimators, beran_models):
        prototype = []
        for iteration in range(n_estimators):
            beran = beran_models[iteration]
            prototype_k = beran.prototype_weights(X_test)
            prototype.append(prototype_k)
        return np.stack(prototype, axis=1)
    
    def _find_attention_H_S_T(self, X, H, S, T, prototype, epsilon, v, gamma):
        alpha = self._ensemble_weights(prototype, X, gamma)

        attention_H = np.einsum('mkn,mk->mn', H, (1-epsilon)*alpha+epsilon*v)
        attention_S = np.einsum('mkn,mk->mn', S, (1-epsilon)*alpha+epsilon*v)
        attention_answer_T = np.einsum('mk,mk->m', T, (1-epsilon)*alpha+epsilon*v)

        return attention_H, attention_S, attention_answer_T

    def _optimisation_C_index_gradient(self, alpha, H, S, T, left, right,
                              epsilon, lr, const_in_div, num_epoch):
        epsilon = torch.tensor(epsilon, device="cpu")  
        v = torch.tensor(np.ones(alpha.shape[1]), requires_grad=True, device="cpu")
        optimizer = torch.optim.Adam([v], lr=lr)

        for iteration in range(num_epoch):
            optimizer.zero_grad()

            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device="cpu")
            v_tensor = v.unsqueeze(0)
            sm = torch.nn.Softmax(dim=1)
            v_tenser_softmax = sm(v_tensor)
            T_tensor = torch.tensor(T, dtype=torch.float32, device="cpu")
            C = (1 - epsilon) * alpha_tensor + v_tenser_softmax*epsilon
            G = torch.sum(C * T_tensor, dim=1)
            G_ij = -G[left] + G[right]
            G_ij = G_ij/const_in_div

            loss = torch.sum(1 / (1 + torch.exp(G_ij)))/left.shape[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_([v], 10)

            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(v.data, nan=0.0)
            
        return v_tenser_softmax.detach().cpu().numpy(), epsilon.item()
    
    def _optimisation_MAE_gradient(self, alpha, T, epsilon, times_train, delta_train, lr, num_epoch):
        epsilon = torch.tensor(epsilon, device="cpu")  
        v = torch.tensor(np.ones(alpha.shape[1]), requires_grad=True, device="cpu")
        optimizer = torch.optim.Adam([v], lr=lr)
        times_train = torch.tensor(times_train, device="cpu")
        for iteration in range(num_epoch):
            optimizer.zero_grad()

            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device="cpu")
            v_tensor = v.unsqueeze(0)
            sm = torch.nn.Softmax(dim=1)
            v_tenser_softmax = sm(v_tensor)
            T_tensor = torch.tensor(T, dtype=torch.float32, device="cpu")
            C = (1 - epsilon) * alpha_tensor + v_tenser_softmax*epsilon
            G = torch.sum(C * T_tensor, dim=1)

            loss = torch.sum(torch.abs((G[delta_train]) - times_train[delta_train]))/times_train[delta_train].shape[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_([v], 1000)

            optimizer.step()
            with torch.no_grad():
                torch.nan_to_num_(v.data, nan=0.0)
        
        return v_tenser_softmax.detach().cpu().numpy(), epsilon.item()
    
    def _optimisation_C_index_linear(self, alpha, T, epsilon, left, right, n_estimators):
        constraints = []
        prob = None

        epsilon = cp.Constant(epsilon) # 1

        v = cp.Variable(n_estimators, nonneg=True)  # k
        constraints.append(cp.sum(v) == 1)
        C = (1-epsilon)*alpha+epsilon*v[np.newaxis,:]   # m*k

        xi = cp.Variable(left.shape[0], nonneg=True)   # power J
        G = cp.sum(cp.multiply(C,T), axis=1)    # m
        G_ij = G[left] - G[right]  # power J
            
        constraints.append(xi >= G_ij)
        prob = cp.Problem(cp.Minimize(cp.mean(xi)), constraints)
        prob.solve(solver=cp.SCS)

        return v.value, epsilon.value
    
    def _optimisation_MAE_linear(self, alpha, T, epsilon, n_estimators, times_train, delta_train):

        constraints = []
        prob = None
        epsilon = cp.Constant(epsilon) # 1
        v = cp.Variable(n_estimators, nonneg=True)  # k
        C = (1-epsilon)*alpha+epsilon*v[np.newaxis,:]   # m*k
        phi = cp.Variable(np.sum(delta_train), nonneg=True)
        U = cp.multiply(C[delta_train] - v[np.newaxis,:]*epsilon,T[delta_train])
        constraints.append(phi - cp.sum(cp.multiply(T[delta_train], v[np.newaxis, :]*epsilon), axis=1) >= cp.sum(U, axis=1) - times_train[delta_train])
        constraints.append(phi + cp.sum(cp.multiply(T[delta_train], v[np.newaxis, :]*epsilon), axis=1) >= -cp.sum(U, axis=1) + times_train[delta_train])
        prob = cp.Problem(cp.Minimize(cp.mean(phi)/cp.max(times_train[delta_train])), constraints)
        prob.solve(solver=cp.SCS)
        return v.value, epsilon.value
    
    def _optimisation(self, alpha, H, S, T, y_train, maximum_number_of_pairs, n_estimators, 
                      c_index_optimisation, MAE_optimisation, times_train, delta_train,
                      epsilon, lr, const_in_div, num_epoch, mode):

        if c_index_optimisation == False and MAE_optimisation == False:
            return np.array([1/n_estimators]*n_estimators), epsilon

        if c_index_optimisation == True:
            left, right = self._find_set_for_C_index_optimisation(y_train, maximum_number_of_pairs)
            if mode == 'gradient':
                v, epsilon = self._optimisation_C_index_gradient(alpha, H, S, T, left, right, 
                                                                epsilon, lr, const_in_div, num_epoch)
            if mode == 'linear':
                v, epsilon = self._optimisation_C_index_linear(alpha, T, epsilon, left, right, n_estimators)
        elif MAE_optimisation == True:
            if mode == 'gradient':
                v, epsilon = self._optimisation_MAE_gradient(alpha, T, epsilon, times_train, delta_train, lr, num_epoch)
            if mode == 'linear':
                v, epsilon = self._optimisation_MAE_linear(alpha, T, epsilon, n_estimators, times_train, delta_train)
            
        return v, epsilon


    def fit(self, X_train, y_train):
        self.X_train, self.y_train, self.times_train, self.delta_train = self._sorting(X_train, y_train)
        self.left, self.right = self._find_set_for_C_index_optimisation(self.y_train, self.maximum_number_of_pairs)
        list_of_kernels = ["epanechnikov", "triangular", "quartic", "gauss"]
        number_of_items_in_subsample = int(self.X_train.shape[0]*self.size_bagging)
        self.cluster_points = self._select_random_points_with_neighbors(self.X_train, self.n_estimators, number_of_items_in_subsample)
        kernels = np.random.choice(list_of_kernels, self.n_estimators)
        self.beran_models = self._train_each_beran(self.X_train, self.y_train, self.n_estimators, self.cluster_points, kernels,
                                                      self.times_train, self.delta_train,
                                                      self.const_in_div, self.c_index_optimisation, self.MAE_optimisation, self.tau)
        H, S, T = self._find_H_S_T(self.X_train, self.times_train, self.X_train, self.beran_models, 
                                   self.n_estimators, self.cluster_points)
        prototype = self._find_prototypes(self.X_train, self.n_estimators, self.beran_models)
        alpha = self._ensemble_weights(prototype, self.X_train, self.omega)
        self.v, self.epsilon = self._optimisation(alpha, H, S, T, self.y_train, number_of_items_in_subsample,
                                                  self.n_estimators, self.c_index_optimisation,
                                                  self.MAE_optimisation, self.times_train, self.delta_train, self.epsilon,
                                                  self.lr, self.const_in_div, self.num_epoch, self.mode)
    
    def predict(self, X_test):
        H, S, T = self._find_H_S_T(self.X_train, self.times_train, X_test, self.beran_models, 
                                   self.n_estimators, self.cluster_points)
        prototype = self._find_prototypes(X_test, self.n_estimators, self.beran_models)
        att_H, att_S, att_T = self._find_attention_H_S_T(X_test, H, S, T, prototype, self.epsilon, self.v,
                                   self.omega)
        return att_H, att_S, att_T
