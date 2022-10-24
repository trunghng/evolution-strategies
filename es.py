from abc import ABC, abstractmethod
from typing import List
import numpy as np
import numpy.linalg as LA


class EvolutionStrategy(ABC):
    '''
    ES abstract class
    '''

    @abstractmethod
    def ask(self):
        pass


    @abstractmethod
    def tell(self):
        pass


    @abstractmethod
    def stop(self):
        pass


    @abstractmethod
    def result(self):
        pass


    @abstractmethod
    def optimize(self):
        pass


class CMAES(EvolutionStrategy):
    '''
    CMA Evolution Strategy
    '''

    DEFAULT_MAXFEVALS = lambda popsize, n: 100 * popsize \
            + 150 * (n + 3)**2 * popsize**0.5
    DEFAULT_POPSIZE = lambda n: 4 + int(3 * np.log(n))
    DEFAULT_PARENT_NUM = lambda popsize: popsize // 2

    def __init__(self, xstart: List[float],
                sigma: float,
                popsize: int=None,
                mu: int=None,
                max_fevals: int=None,
                ftarget: float=None) -> None:
        '''
        Parameters
        ----------
        xstart: initial point, its length also decides
            the number of dimensions of the search space
        sigma: initial step-size
        popsize: population size
        mu: number of parents/selected points
        max_fevals: maximum number of evaluations
        ftarget: target of fitness value
        '''
        # Set trategy parameters
        n = len(xstart)
        self.n = n
        self.sigma = sigma
        self.lambda_ = popsize if popsize else CMAES.DEFAULT_POPSIZE(n)
        self.mu = mu if mu else CMAES.DEFAULT_PARENT_NUM(self.lambda_)
        self.max_fevals = max_fevals if max_fevals \
                else CMAES.DEFAULT_MAXFEVALS(self.lambda_, n)
        self.ftarget = ftarget

        ## Set recombination weights
        _weights = np.array([np.log((self.lambda_ + 1) / 2) - np.log(i + 1) 
                if i < self.mu else 0 for i in range(self.lambda_)], dtype=np.float64)
        self.weights = _weights / np.sum(_weights[:self.mu])
        self.weights_pos = self.weights[:self.mu]
        self.weights_neg = self.weights[self.mu:]
        self.mueff = np.sum(self.weights_pos)**2 / np.sum(self.weights_pos**2)

        ## Set adaptation parameters
        alpha_cov = 2
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.c1 = alpha_cov / ((n + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, alpha_cov * (self.mueff - 2 
                + 1 / self.mueff) / ((n + 2)**2 + alpha_cov * self.mueff / 2))
        self.csigma = (self.mueff + 2) / (n + self.mueff + 5)
        # self.dsigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) 
        #         / (n + 1)) - 1) + self.csigma
        self.dsigma = 2 * self.mueff / self.lambda_ + 0.3 + self.csigma

        # Initialization
        self.psigma = np.zeros(n)
        self.pc = np.zeros(n)
        self.C = np.identity(n)
        self.xmean = np.array(xstart)
        self.sigma = sigma
        self.count_eval = 0


    def ask(self):
        '''
        Sample @self.lambda_ offsprings: x ~ m + sigma * N(0, C)
        -------------------------
        X, Y, Z in R^{@self.n x @self.lambda_}
        Z ~ N(0, I)
        Y = Q.Lambda.Z ~ N(0, C)
        X = m + sigma * Y ~ m + sigma * N(0, C)
        '''
        self._diagonalize_C()
        Z = np.random.randn(self.n, self.lambda_)
        Y = LA.multi_dot([self.Q, self.Lambda**0.5, Z])
        X = self.xmean.reshape(self.n, 1) + self.sigma * Y
        return X


    def tell(self, X: np.ndarray, fitness_list: np.ndarray):
        self.count_eval += self.lambda_

        # Selection & Recombination
        ordered_indices = np.argsort(fitness_list)
        self.fitness_list = np.sort(fitness_list)
        self.X = X[:,ordered_indices]
        xmean_old = self.xmean.copy()
        Y = (self.X - xmean_old.reshape(self.n, 1)) / self.sigma
        self.xmean = np.sum(self.weights_pos * self.X[:,:self.mu], axis=1)

        # Update evolution paths: pc, psigma
        step = self.xmean - xmean_old

        ## Update psigma
        ### psigma normalized constant
        psigma_nc = np.sqrt(self.csigma * (2 - self.csigma) * self.mueff) / self.sigma
        self.psigma = (1 - self.csigma) * self.psigma + psigma_nc * self.C_invsqrt.dot(step)

        ## Update pc
        g = self.count_eval / self.lambda_
        hsigma = (np.sum(self.psigma**2) / (self.n * (1 - (1 - self.csigma)**(2 * g)))) \
            < 2 + 4. / (self.n + 1)
        ### pc normalized constant
        pc_nc = np.sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma
        self.pc = (1 - self.cc) * self.pc + hsigma * pc_nc * step

        # Covariance matrix adaption
        # C = (1 - c1 * (1 - hsigma) * cc * (2 - cc) - c1 - cmu * sum(weights))
        #       + c1 * pc.pc^T + cmu * sum(wi * yi.yi^T)
        r1_upd = self.c1 * self._xxT(self.pc)
        
        elite_cov = np.sum(np.array([self.weights_pos[i] * self._xxT(Y[:,:self.mu][:,i]) \
            for i in range(len(self.weights_pos))]), axis=0)
        non_elite_cov = np.sum(np.array([self.weights_neg[i] * self.n / ((np.sum((self.C_invsqrt.dot(Y[:,self.mu:][:,i]))**2))**0.5) \
            * self._xxT(Y[:,self.mu:][:,i]) for i in range(len(self.weights_neg))]), axis=0)
        rmu_upd = self.cmu * (elite_cov + non_elite_cov)

        
        delta_hsigma = (1 - hsigma) * self.cc * (2 - self.cc)
        self.C = (1 - self.c1 * delta_hsigma - self.c1 - self.cmu * np.sum(self.weights)) * self.C \
            + r1_upd + rmu_upd

        # Step-size adaption
        self.sigma *= np.exp(min(1, self.csigma / (self.dsigma * 2) * (np.sum(self.psigma**2) / self.n - 1)))


    def stop(self):
        term_result = {}
        if self.count_eval >= self.max_fevals:
            term_result['max_fevals'] = self.max_fevals
        if self.condition_number > 1e14:
            term_result['condition_cov'] = self.condition_number
        if self.sigma * max(self.C_eigenvals)**0.5 < 1e-15:
            term_result['tol_x'] = 1e-15
        if len(self.fitness_list) > 1 and \
            self.fitness_list[-1] - self.fitness_list[0] < 1e-12:
            term_result['tol_fun'] = 1e-12
        if self.ftarget is not None and len(self.fitness_list) > 0 \
                and self.fitness_list[0] <= self.ftarget:
            term_result['ftarget'] = self.ftarget
        return term_result


    def result(self):
        best_idx = np.argmin(self.fitness_list)
        
        try:
            if self.fitness_list[best_idx] < self.best_val:
                self.best_sol = self.X[:, best_idx]
                self.best_val = self.fitness_list[best_idx]
        except AttributeError:
            self.best_sol = self.X[:, best_idx]
            self.best_val = self.fitness_list[best_idx]

        return {
            'best_sol': self.X[:, best_idx],
            'best_val': self.fitness_list[best_idx],
            'solutions': self.X,
            'count_eval': self.count_eval,
            'count_iter': self.count_eval / self.lambda_
        }


    def optimize(self):
        pass


    def _diagonalize_C(self):
        self.C_eigenvals, self.Q  = LA.eigh(self.C)
        self.Lambda = np.diag(self.C_eigenvals)
        assert np.min(self.C_eigenvals) > 0, f'Covariance matrix is not PD!: {min(self.C_eigenvals)}'
        self.condition_number = np.max(self.C_eigenvals) / np.min(self.C_eigenvals) \
            if np.min(self.C_eigenvals) > 0 else 0
        self.C_invsqrt = LA.multi_dot([self.Q, LA.inv(np.sqrt(self.Lambda)), self.Q.T])


    def _xxT(self, x: np.ndarray):
        return x.reshape(x.shape[0], 1).dot(x.reshape(1, x.shape[0]))


class xNES(EvolutionStrategy):


    def __init__(self) -> None:
        pass



