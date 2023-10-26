import numpy as np
import cvxpy as cp


class DiscreteFairFront:
    def __init__(self, A, C, mu_SY, dist_inf, alpha_SP=1.0, alpha_EO=1.0, alpha_OAE=1.0):
        """
        :A: number of sub-groups
        :C: number of classes
        :mu_SY: joint probability distribution of (S,Y)

        :InputDist: input probability distribution to compute expectation
        :dist_inf: transition matrix from (S,Y) to X & mapping from index set to support of X (required if InputDist==True)

        :alpha_SP: fairness threshold for statistical parity
        :alpha_EO: fairness threshold for equalized odds
        :alpha_OAE: fairness threshold for overall accuracy equality
        """
        
        self.A = A
        self.C = C
        self.mu_SY = mu_SY
        self.mu_S = np.sum(self.mu_SY, axis = 1)

        self.T_SY_X = dist_inf["T"]
        self.map_ind_X = dist_inf["map"]
        self.calX = len(self.map_ind_X)

        self.alpha_SP = alpha_SP
        self.alpha_EO = alpha_EO
        self.alpha_OAE = alpha_OAE

        
    
    def SP_constraint(self, s1, s2, y_hat, P):
        return cp.abs(cp.sum([((self.mu_SY[s1][y]/self.mu_S[s1])*P[s1*self.C + y][y_hat] \
                               - (self.mu_SY[s2][y]/self.mu_S[s2])*P[s2*self.C + y][y_hat]) for y in range(self.C)]))

    def EO_constraint(self, s1, s2, y, y_hat, P):
        return cp.abs(P[s1*self.C + y][y_hat] - P[s2*self.C + y][y_hat])

    def OAE_constraint(self, s1, s2, P):
        return cp.abs(cp.sum([((self.mu_SY[s1][y]/self.mu_S[s1])*P[s1*self.C + y][y] - (self.mu_SY[s2][y]/self.mu_S[s2])*P[s2*self.C + y][y]) for y in range(self.C)]))


        
    def FairFront(self):
        
        # transition matrix from x to Yhat
        H = cp.Variable((self.calX, self.C), nonneg=True)
        # transition matrix from (s,y) to Yhat
        P = cp.Variable((self.A * self.C, self.C), nonneg=True)
        
        # initialize constraints list
        constraints1 = []
        
        # constraints for H and P to be valid transition matrices
        for i in range(self.calX):
            constraints1 += [cp.sum(H[i,:]) == 1]   
        for i in range(self.A * self.C):
            constraints1 += [cp.sum(P[i,:]) == 1]
        
        # impose Markov chain conditions
        for i in range(self.A * self.C):
            for j in range(self.C):
                constraints1 += [P[i,j] == (self.T_SY_X[i,:] @ H[:,j])]
        

        # There are  A * (A-1)/2 * C constraints in SP
        # There are A * (A-1)/2 * C * C constraints in EO
        # There are  A * (A-1)/2 constraints in OAE
        for i in range(self.A):
            for j in range(i+1, self.A):
                # add OAE contrainst
                constraints1 += [ self.OAE_constraint(i, j, P) <= self.alpha_OAE] 
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
        
        # Form objective
        obj = cp.Maximize(obj_sum)
        # Form and solve problem.
        prob = cp.Problem(obj, constraints1)
        prob.solve(solver=cp.ECOS)
        
        
        # solved_p = P.value
        # Hv = H.value
        
        return prob.value



