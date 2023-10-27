import numpy as np
import cvxpy as cp
import dccp


class FairFront:
    def __init__(self, A, C, g, InputDist, sensitive_s, label_y, dist_inf = [], data_inf = [],max_iterations = 100, num_cut = 6, alpha_SP=1.0, alpha_EO=1.0, alpha_OAE=1.0):
        """
        :A: number of sub-groups
        :C: number of classes
        :mu_SY: joint probability distribution of (S,Y)
        :g: Pr(S,Y|X)
        :InputDist: input probability distribution to compute expectation
        :sensitive_s: name of the sensitive attribute column
        :label_y: name of the label column
        :dist_inf: transition matrix from (S,Y) to X & mapping from index set to support of X (required if InputDist==True)
        :data_inf: number of data & dataset & estimation error bound (required if InputDist==False)
        :max_iterations: max number of iterations of greedy improvement algorithm
        :num_cut: number of cuts
        :alpha_SP: fairness threshold for statistical parity
        :alpha_EO: fairness threshold for equalized odds
        :alpha_OAE: fairness threshold for overall accuracy equality
        """
        self.A = A
        self.C = C
        self.g = g
        self.InputDist = InputDist
        self.max_iterations = max_iterations
        self.num_cut = num_cut
        self.alpha_SP = alpha_SP
        self.alpha_EO = alpha_EO
        self.alpha_OAE = alpha_OAE
        self.sensitive_s = sensitive_s
        self.label_y = label_y
        
        
        #print("Using input distribution?", self.InputDist)
        if self.InputDist == True:
            self.T_SY_X = dist_inf["T"]
            self.map_ind_X = dist_inf["map"]
            self.calX = len(self.map_ind_X)
            self.mu_SY = dist_inf["mu_SY"] # mu_SY if pre-specified S,Y joint distribution 
            
            # marginal distribution of X
            self.P_X = np.zeros(self.calX)
            for i in range(self.calX):
                self.P_X[i] = self.mu_SY.flatten() @ self.T_SY_X[:,i]
            
        else:
            self.N = data_inf["num_data"]
            self.original_df = data_inf["original_df"] #original dataframe 
            self.data_x = data_inf["data_x"]
            self.est_error_bound = data_inf["est_error_bound"]

            # get the mu_s,y values, encoded in matrix mu_SY
            mu_SY = np.zeros((self.A, self.C))
            mu_SY = np.zeros((2, 2))

            for index, row in self.original_df.iterrows():
                s,y = int(row[self.sensitive_s]), int(row[self.label_y])
                mu_SY[s][y] = mu_SY[s][y] + 1
            self.mu_SY = mu_SY/self.N

        self.compute_marginal_stats() 
    
    def compute_marginal_stats(self):
        """
        # compute marginal statistics
        """
        
    
        # compute marginal distribution of mu_S
        self.mu_S = np.sum(self.mu_SY, axis = 1)

        # compute Lambda_mu
        self.Lambda_mu = np.zeros((self.A * self.C, self.A * self.C))
        np.fill_diagonal(self.Lambda_mu, np.hstack(self.mu_SY))
        
        
    
    def SP_constraint(self, s1, s2, y_hat, P):
        return cp.abs(cp.sum([((self.mu_SY[s1][y]/self.mu_S[s1])*P[s1*self.C + y][y_hat] \
                               - (self.mu_SY[s2][y]/self.mu_S[s2])*P[s2*self.C + y][y_hat]) for y in range(self.C)]))

    def EO_constraint(self, s1, s2, y, y_hat, P):
        return cp.abs(P[s1*self.C + y][y_hat] - P[s2*self.C + y][y_hat])

    def OAE_constraint(self, s1, s2, P):
        return cp.abs(cp.sum([((self.mu_SY[s1][y]/self.mu_S[s1])*P[s1*self.C + y][y] - (self.mu_SY[s2][y]/self.mu_S[s2])*P[s2*self.C + y][y]) for y in range(self.C)]))
    
    
    
    def LHS(self, cut, P_matrix):
        LHS_sum = 0
        for j in range(self.C):
            LHS_sum += cp.max(cp.vstack([cut[:,i].T @ self.Lambda_mu @ P_matrix[:,j] \
                            for i in range(self.num_cut)]))
        return LHS_sum
    
    
    
    def RHS_emp(self, cut):
        
        assert callable(self.g)
        RHS_sum = 0
        for j in range(self.calX):
            xk = self.map_ind_X[j]
            RHS_sum += np.max([cut[:,i].T @ self.g(xk) for i in range(self.num_cut)]) * self.P_X[j]
        
        return RHS_sum
    
    def RHS_emp_var(self, cut):
        '''
        Used when distribution pre-specified 
        '''
        assert callable(self.g)
        RHS_sum = 0
        for j in range(self.calX):
            xk =self.map_ind_X[j]
            RHS_sum += cp.max(cp.vstack([cut[:,i].T @ self.g(xk) for i in range(self.num_cut)])) * self.P_X[j]
            
        return RHS_sum
    
    
    
    def RHS_data(self, cut):
        assert callable(self.g)
        RHS_sum = 0
        for k in range(self.N):
            xk = self.data_x[k]
            RHS_sum += np.max([cut[:,i].T @ self.g(xk) for i in range(self.num_cut)])
        return RHS_sum/self.N
    
    
    def RHS_data_var(self, cut):
        assert callable(self.g) 
        RHS_sum = 0
        for k in range(self.N):
            xk = self.data_x[k]
            RHS_sum += cp.max(cp.vstack([cut[:,i].T @ self.g(xk) for i in range(self.num_cut)]))
        return RHS_sum/self.N
        





    def step_1(self, A_list):
        '''
        # step 1 of the algorithm---optimize for P
        :A_list: list of current cuts
        '''
        
        # initialize variable
        # each element of P is non-negative
        P = cp.Variable((self.A * self.C, self.C), nonneg=True) 
        
    
        # initialize constraints list
        constraints1 = []
    
        # constraints for the random P (rows sum to 1 and each element non-negative)
        # P is AC * C
        for i in range(self.A * self.C):
            constraints1 += [cp.sum(P[i,:]) == 1]   
            
        if len(A_list) > 0:
            for cut in A_list:
                if self.InputDist == True:
                    constraints1 += [self.LHS(cut, P) <= self.RHS_emp(cut)]
                else:
                    constraints1 += [self.LHS(cut, P) <= self.RHS_data(cut) + self.est_error_bound]
        
        
        # There are  A * (A-1)/2 * C constraints in SP
        # There are A * (A-1)/2 * C * C constraints in EO
        # There are  A * (A-1)/2 constraints in OAE
        for i in range(self.A):
            for j in range(i+1, self.A):
                # add OAE contrainst
                constraints1 += [ self.OAE_constraint(i, j, P) <= self.alpha_OAE] # P_prime is our AC*C example
                for k in range(self.C):
                    # add SP constraints
                    constraints1 += [ self.SP_constraint(i, j, k, P) <= self.alpha_SP]
                    for q in range(self.C):
                        # add EO constraint 
                        constraints1 += [ self.EO_constraint(i, j, k, q, P) <= self.alpha_EO] 
    
        
        # define the overall objective sum 
        obj_sum = 0
        for s in range(self.A):
            for y in range(self.C):
                obj_sum += self.mu_SY[s][y] * P[s*self.C + y][y]
        obj_sum.value
         
        # Form objective.
        obj = cp.Maximize(obj_sum)
        # Form and solve problem.
        prob = cp.Problem(obj, constraints1)
        sol = prob.solve(solver=cp.ECOS)
        return sol, P.value
    
        
    def step_2(self, current_P):
        '''
        # step 2 of the algorithm---generate optimal cuts
        current P: current transition matrix
        '''
        
        assert callable(self.g)
    
        # initialize variables to optimize over
        an = cp.Variable((self.A*self.C, self.num_cut))
        t = cp.Variable(1)
    
        # constraints for an, all entries have to be between -1 and 1
        constraints2 = []
        for i in range(self.A*self.C):
            for j in range(self.num_cut):
                constraints2 += [cp.abs(an[i][j]) <= 1]
                
        if self.InputDist == True:
            constraints2 += [t >= self.RHS_emp_var(an)]
        else:
            constraints2 += [t >= self.RHS_data_var(an)]
        
        # Form objective.
        obj_diff = t - self.LHS(an, current_P)
        obj_part2 = cp.Minimize(obj_diff)
        
        # Form and solve problem.
        prob2 = cp.Problem(obj_part2, constraints2)
        result = prob2.solve(method='dccp')
        
    
        # return both the minimized value and the cut 
        return result[0], an.value
    
    
    def Algorithm(self):
        """
        # greedy improvement algorithm for solving fairness-accuracy trade-offs
        """
        
        A_list = []
        P_list = []

        #print("algorithm starts")
        for t in range(self.max_iterations):
            # if t % 50 == 0:
            #     print("current iteration:", t)
            sol, current_P = self.step_1(A_list)
            #print("passed step 1, current P is", current_P)
            
            # step 2 return that current_difference, new_cut  
            current_value, new_cut = self.step_2(current_P)
            #print("passed step 2")
        
            
            A_list.append(new_cut)
            P_list.append(current_P)

        return sol, P_list, A_list