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
    DEFAULT_POPSIZE = lambda n: 4 + int(3 * np.log(n))
    DEFAULT_PARENT_NUM = lambda lambda_: lambda_ // 2

    def __init__(self, xstart: List[float],
                sigma: float,
                popsize: int=None,
                mu: int=None) -> None:
        '''
        Parameters
        ----------
        xstart: initial point, its length also decides
            the number of dimensions of the search space
        sigma: initial step-size
        popsize: population size
        mu: number of parents/selected points
        '''
        # Set trategy parameters
        n = len(xstart)
        self.n = n
        self.sigma = sigma
        self.lambda_ = popsize if popsize else DEFAULT_POPSIZE(n)
        self.mu = mu if mu else DEFAULT_PARENT_NUM(self.lambda_)
        self.chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))

        ## Set recombination weights
        _weights = np.array([np.log((self.lambda_ + 1) / 2) - np.log(i + 1) 
                for i in range(self.lambda_)])
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
        self.dsigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) 
                / (n + 1)) - 1) + self.csigma
        

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
        self.condition_number = matrix.condition_number(self.C)
        self.Q, self.Lambda = matrix.diagonalize(self.C)
        Z = np.random.randn(self.n, self.lambda_)
        self._diagonalize_C()
        Y = LA.multi_dot(self.Q, Lambda, self.Z)
        X = self.xmean + self.sigma * Y
        return X


    def tell(self, X: np.ndarray, fitness_vals: np.ndarray):
        self.count_eval += self.lambda_

        # Selection & Recombination
        ordered_indices = np.argsort(fitness_vals)
        elite_indices = ordered_indices[:self.mu]
        non_elite_indices = ordered_indices[self.mu:]
        elites = X[:, elite_indices]
        non_elites = X[:]
        xmean_old = self.xmean.copy()
        self.xmean = np.sum(self.weights_pos * elites, axis=1)

        # Update evolution paths: pc, psigma
        step = self.xmean - xmean_old

        ## Update psigma
        ### psigma normalized constant
        psigma_nc = np.sqrt(self.csigma * (2 - self.csigma) * self.mueff) / self.sigma
        self.psigma = (1 - self.csigma) * self.psigma + psigma_nc * self.C_invsqrt.dot(step)

        ## Update pc
        g = self.count_eval / self.lambda_
        psigma_len = LA.norm(self.psigma)
        hsigma = (psigma_len / np.sqrt(self.n * (1 - (1 - self.csigma)**(2 * g)))) \
            < 1.4 + 2 / (self.n + 1)
        ### pc normalized constant
        pc_nc = np.sqrt(self.cc * (2 - self.cc) * self.mueff) / self.psigma
        self.pc = (1 - self.cc) * self.pc + hsigma * pc_nc * step

        # Step-size adaption
        self.sigma *= np.exp(min(1, self.csigma / self.dsigma * (psigma_len**2 / self.n - 1) / 2))

        # Covariance matrix adaption
        r1_upd = self.c1 * self,_xxT(self.pc)
        elite_cov = np.sum(np.array([self.weights_pos[i] * self._xxT(elites[:, i] - xmean_old) \
            for i in range(len(self.weights_pos))]), axis=0)
        non_elite_cov = np.sum(np.array([self.weights_neg[i] * self.n / (np.linalg.norm( \
            self.C_invsqrt.dot(non_elites[:, i] - xmean_old))**2) * self._xxT(non_elites[:, i] - xmean_old) \
            for i in range(len(self.weights_neg))]), axis=0)
        rmu_upd = elite_cov + non_elite_cov
        delta_hsigma = (1 - hsigma) * self.cc * (2 - self.cc)
        self.C = (1 - self.c1 * delta_hsigma - self.c1 - self.cmu * np.sum(self.weights)) * self.C \
            + r1_upd + rmu_upd


    def stop(self):
        pass


    def result(self):
        pass


    def optimize(self):
        pass


    def _diagonalize_C(self):
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        self.Q, eigenvals = LA.eig(self.C)
        assert np.min(eigenvals) > 0, 'Covariance matrix is not PD!'
        self.Lambda = np.diag(eigenvals)
        self.kappa_C = np.max(eigenvals) / np.min(eigenvals)
        self.C_invsqrt = np.copy(self.C)
        self.C_invsqrt = LA.multi_dot(self.Q, 1 / np.sqrt(self.Lambda), self.Q.T)


    def _xxT(self, x: np.ndarray):
        return x.reshape(x.shape[0], 1).dot(x.reshape(1, x.shape[0]))

