import numpy as np
import scipy as sp
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

class Branching_Process:
    
    def __init__(self, offspring, time = np.arange(100), offspring_params = None):
        self.offspring = offspring
        self.time = time
        self.N = len(time)
        self.time_step = time[1] - time[0]
        #assert (np.diff(time) == self.time_step).all(), "Need to have uniform time steps"
        self.offspring_params = offspring_params
        self.infectiousness_profile = None
        self.cutoff = None
        self.cutoff_idx = None
        
        
    def vectorised_integral_equation(self):
        raise NotImplementedError
        
    def lam(self, tau):
        raise NotImplementedError
        
    def set_cutoff(self, cutoff):
        
        self.cutoff = cutoff 
        self.cutoff_idx = int(cutoff / self.time_step)
        assert(self.cutoff_idx <= self.N), "Cutoff needs to be less than the maximum time specified in the Branching Process"

    
    def set_lifetime_distribution(self, distribution = 'exp', lamb = 1):
        time = self.time
        if distribution == 'exp':
            self.lifetime_distribution = stat.expon
        else:
            
            self.lifetime_distribution = distribution
    
    def set_lifetime_pdf(self, distribution):
        time = self.time
        
        self.lifetime_pdf = distribution
        
            
    def set_infectiousness_profile(self, distribution):
        time = self.time

        self.infectiousness_profile = distribution
        
            
    def set_transmission_rate(self, rate):
        time = self.time
        self.transmission_rate = rate
        
    def set_immigration(self, immigration):
        immigration = immigration
    
    
        
    def pgf_vec(self, s, count = "prev", return_diag=True):
        self.pgf_prep()
        
        
        N, offspring, Gbar, = self.N, self.offspring, self.Gbar
        DG, DV = self.lifetime_gradient, self.infectiousness_gradient
        
        F = np.zeros((N+1, N+1), dtype=np.complex_)
        F[:, 0] = s
        for i in range(1,N):
            F[i:(N+1), i] = self.vectorised_integral_equation(i, s, count, F)

        
        if return_diag:
            return np.diag(F)
        else:
            return F
        
   
    # Calculate probability of extinction from probability generating function - either vec or direct function
    def extinction_probability(self, pgf_func):
        return np.real_if_close(pgf_func(0))
        

        

    def pgf_immigration(self, s, immigration, count = 'prev'):
        F = self.pgf_vec(s, count = "prev", return_diag=False) # Count needs to be prev for immigration (check literature!)
        N, time_step = self.N, self.time_step
        
        D_t = np.triu(np.tile(np.arange(N, -1, -1) * time_step, (N, 1)))
        D_tau = np.triu(np.tile(np.arange(0, N) * time_step, (N+1, 1)).T)
        
        immigration_tau = immigration(D_t) 
        
        R =  np.zeros((N+1, N+1), dtype=np.complex_)
        R[:, 0] = 1
        
        for i in range(1, N+1):
        #     R[i:(N + 1), i] = (np.sum(immigration_tau[0:(N - i + 1), 
        #                            (N - i):N]*R[i:(N + 1), 0:i]*F[i:(N + 1), 0:i], axis = 1))
            
            R[i:(N+1), i] = (np.sum(immigration_tau[0:(N - i + 1), 
                                    (N - i):N]*(F[i:(N + 1), 0:i] - 1), axis = 1))
        
        return np.exp(np.diag(R))
        
    def analytic_mean(self):
        N, time_step, lifetime_distribution = self.N, self.time_step, self.lifetime_distribution
        infectiousness_profile, transmission_rate = self.infectiousness_profile, self.transmission_rate
        
        def h(t, tau):
            return 1 - lifetime_distribution(t - tau, tau)

        fhat = np.zeros(shape = (N + 1, N + 1))

        for n in range(N + 1):
            for i in range(n + 1):
                if i == 0:
                    fhat[n, i] = h(n * time_step, n * time_step)
                else:
                    fhat[n, i] = h(n * time_step, (n - i) * time_step) + np.sum(fhat[n, 
                                i - np.arange(1, i + 1)] * 
                                self.lam(time_step * np.arange(1, i + 1), (n - i) * time_step) )
                    
        return np.diagonal(fhat)
    
    def Malthusian_growth_rate(self, mean_offspring, x0 = 0.1, t_inf = np.linspace(0, 10000, num=1000)):
        
        lifetime_pdf = self.lifetime_pdf
        
        def Malthusian_integrand(self, u, t, alpha, mean_offspring, lifetime_pdf):
            return np.exp(-alpha*t) * lifetime_pdf(t, 0) * mean_offspring

        def Malthusian_parameter(self, alpha, mean_offspring = 1.5, t_inf = t_inf):
            return (sp.integrate.odeint(Malthusian_integrand, 0, 1000, 
                                args = (alpha, mean_offspring)) - 1)[-1]

        malpha = sp.optimize.fsolve(Malthusian_parameter, 0.04, args = (1.3, t_inf))
        return malpha

    
class cmj(Branching_Process):
    
    
    def calculate_Gbar(self, t, tau):
        lifetime_distribution = self.lifetime_distribution
        return 1 - lifetime_distribution(t - tau, tau)
    
    def calculate_DV(self, D_t, D_tau):
        N = self.N
        infectiousness_profile = self.infectiousness_profile
        transmission_rate = self.transmission_rate
        DV = -transmission_rate(D_t[:, :-1] + D_tau[:, 
                                    :-1]) * np.diff(infectiousness_profile(D_t))
        return DV
    
    def pgf_prep(self):   
        N = self.N
        time_step = self.time_step
        time = self.time
        lifetime_distribution = self.lifetime_distribution
        transmission_rate = self.transmission_rate
        infectiousness_profile = self.infectiousness_profile
        offspring = self.offspring
        
        Gbar_t = np.tril(np.tile(np.arange(0, N+1) * time_step, (N+1, 1)).T)  
        Gbar_tau = np.tril(toeplitz(np.arange(0, N+1) * time_step))  
        Gbar = self.calculate_Gbar(Gbar_t, Gbar_tau)
        
        D_t = np.triu(np.tile(np.arange(N, -1, -1) * time_step, (N, 1)))
        D_tau = np.triu(np.tile(np.arange(0, N) * time_step, (N+1, 1)).T)
        
        DV = -transmission_rate(D_t[:, :-1] + D_tau[:, :-1]) * np.diff(infectiousness_profile(D_t))
        DG = -np.diff(lifetime_distribution(D_t, D_tau))
        
        self.Gbar = Gbar
        self.lifetime_gradient, self.infectiousness_gradient = DG, DV
    
        
    def vectorised_integral_equation(self, i, s, count, F):
        
        assert(count=="prev" or count == "ci"), "Count needs to be either 'prev' (for prevalence) or 'ci' (for cumulative incidence)."

        N, offspring, Gbar, = self.N, self.offspring, self.Gbar
        DG, DV = self.lifetime_gradient, self.infectiousness_gradient
        
        cutoff_idx = self.cutoff_idx
        cutoff = self.cutoff
        
        def q_1(z, l):    
            return l * np.exp(z)

        if count == "prev":
            def q_2(z, l):
                return np.exp(z)
        if count == "ci":
            def q_2(z, l):
                return l * np.exp(z)
       
    
        if cutoff_idx is not None:
            end_idx = int(np.min((cutoff_idx, i)))
        else:
            end_idx = i
        # Correct code for full
        #B = offspring(F[i:(N + 1), 0:i]) * DV[0:(N - i + 1), (N - i):N]
        #B_1 = np.cumsum(B[:, ::-1], axis = 1)[:, ::-1]

        #int_1 = Gbar[i:(N + 1), i] * q_1(B_1[:, 0], s)

        #B_2 = np.concatenate((B_1[:, 1:i], np.zeros((B_1.shape[0], 1))), axis = 1)
        #int_2 = np.sum(q_2(B_2, s) * DG[0:(N - i + 1), (N - i):N], axis = 1)
        
        
        
        B = offspring(F[i:(N + 1), 0:end_idx]) * DV[0:(N - i + 1), (N - end_idx):N]
        B_1 = np.cumsum(B[:, ::-1], axis = 1)[:, ::-1]
        int_1 = Gbar[i:(N + 1), i] * q_1(B_1[:, 0], s)

        B_2 = np.concatenate((B_1[:, 1:i], np.zeros((B_1.shape[0], 1))), axis = 1)
        int_2 = np.sum(q_2(B_2, s) * DG[0:(N - i + 1), (N-end_idx):N], axis = 1)
        
        integral = int_1 + int_2
        return integral
    
    def lam(self, t, tau):
        
        time_step, lifetime_distribution = self.time_step, self.lifetime_distribution
        infectiousness_profile, transmission_rate = self.infectiousness_profile, self.transmission_rate
        
        return transmission_rate(t + tau) * (infectiousness_profile(t) -
                infectiousness_profile(t-time_step)) * (1 - lifetime_distribution(t, tau))
    
    def decompose_variance(self, R, phi ,prevalence=True):
        N = self.N
        time = self.time
        G = self.lifetime_distribution(time, 0)
        v = self.infectiousness_profile(time)
        g = np.append(np.diff(G), 0)
        plt.plot(time, g)
        plt.title('Lifetime Distribution')
        V = np.zeros((N, N))
        V_it = np.zeros((N, N))
        V_ni = np.zeros((N, N))

        F = np.zeros((N, N))
        store1 = np.zeros((N, N))
        store2 = np.zeros((N, N))
        store3 = np.zeros((N, N))
        F[:, 0] = G[0]
        V[:, 0] = 0
        V_it[:, 0] = 0
        V_ni[:, 0] = 0
        EYsq = 1+(1/phi)
        for c in range(0, N):
            for t in range(1, c+1):
                convolution = 0
                term1=0 # G
                term2=0 # 2GM
                term3=0 # G M^2 
                term4=0 # Int Sg
                term5=0 # V
                term6=0 # MY
                term7=0 # -M^2
                term2_integral=0
                term3_integral=0
                convolutionS = 0
                term5_it=0 # V
                term5_ni=0 # V            
                for u in range(0, t+1):
                    convolution +=  R[c-t+u] * G[u] * v[u] * F[c, t-u]
                if prevalence:
                    F[c, t] = G[t] + convolution
                else:
                    F[c, t] = 1 + convolution            
                for u in range(0,t+1):
                    term4_integral = 0
                    if prevalence:
                        term2_integral += R[c-t+u] * v[u] * F[c, t-u] 
                    else:
                        term2_integral += R[c-t+u] * v[u] * F[c, t-u] * G[u]               
                    term3_integral += R[c-t+u] * v[u] * F[c, t-u] 
                    term5 += R[c-t+u] * G[u] * v[u] * V[c, t-u]
                    term5_it += R[c-t+u] * G[u] * v[u] * V_it[c, t-u]
                    term5_ni += R[c-t+u] * G[u] * v[u] * V_ni[c, t-u]
                    term6 += R[c-t+u] * G[u] * v[u] * np.square(F[c, t-u]) * EYsq
                    for q in range(0,u):
                        term4_integral += R[c-t+q] * v[q] * F[c, t-q]
                    term4 += np.square(term4_integral)*g[u]
                if prevalence:
                    term1 = G[t]
                    term2 = 2 * G[t] * term2_integral
                    term3 = np.square(term2_integral)*G[t]
                else:
                    term1 = 1
                    term2 = 2 * term2_integral
                    term3 = np.square(term3_integral)*G[t]                
                term7 = np.square(F[c, t])
                V[c,t] = term1 + term2 + term3 + term4 + term5 + term6 - term7
                V_it[c,t] = term1 + term2 + term3 + term4 + term5_it - term7
                V_ni[c,t] = term5_ni + term6
                V[c,t] = term1 + term2 + term3 + term4 + term5 + term6 - term7
                store1[c,t] = term1 + term4 + term2 + term3  - term7
                store2[c,t] = term5
                store3[c,t] = term6 

        output = np.zeros((N,7))
        output[:,0]= np.diag(F)     
        output[:,1]= np.diag(V) 
        output[:,2]= np.diag(store1) 
        output[:,3]= np.diag(store2) 
        output[:,4]= np.diag(store3) 
        output[:,5]= np.diag(V_it)
        output[:,6]= np.diag(V_ni)
        return output      
    
    
class BellmanHarris(Branching_Process):    
    
    def calculate_Gbar(self, t, tau):
        lifetime_distribution = self.lifetime_distribution
        return 1 - lifetime_distribution(t - tau, tau)
    
    def calculate_DV(self, D_t, D_tau):
        transmission_rate = self.transmission_rate
        DV = transmission_rate(D_t[:, :-1] + D_tau[:, 
                                    :-1])
        return DV
    
    def pgf_prep(self):   
        N = self.N
        time_step = self.time_step
        time = self.time
        lifetime_distribution = self.lifetime_distribution
        transmission_rate = self.transmission_rate
        offspring = self.offspring
        
        Gbar_t = np.tril(np.tile(np.arange(0, N+1) * time_step, (N+1, 1)).T)  
        Gbar_tau = np.tril(toeplitz(np.arange(0, N+1) * time_step))  
        Gbar = self.calculate_Gbar(Gbar_t, Gbar_tau)
        
        D_t = np.triu(np.tile(np.arange(N, -1, -1) * time_step, (N, 1)))
        D_tau = np.triu(np.tile(np.arange(0, N) * time_step, (N+1, 1)).T)
        
        DV = self.calculate_DV(D_t, D_tau)
        DG = -np.diff(lifetime_distribution(D_t, D_tau))
        
        self.Gbar = Gbar
        self.lifetime_gradient, self.infectiousness_gradient = DG, DV
    
    def vectorised_integral_equation(self, i, s, count, F):
        
        assert(count=="prev" or count == "ci"), "Count needs to be either 'prev' (for prevalence) or 'ci' (for cumulative incidence)."

        N, offspring, Gbar, = self.N, self.offspring, self.Gbar
        DG, DV = self.lifetime_gradient, self.infectiousness_gradient
        
        cutoff_idx = self.cutoff_idx
        cutoff = self.cutoff
        
        def q_1(z, l):    
            return l

        if count == "prev":
            def q_2(z, l):
                return z#np.exp(z)
        if count == "ci":
            def q_2(z, l):
                return l * z# np.exp(z)
       
    
        if cutoff_idx is not None:
            end_idx = int(np.min((cutoff_idx, i)))
        else:
            end_idx = i
       

        B = offspring(F[i:(N + 1), 0:end_idx]) * DV
        #B_1 = np.cumsum(B[:, ::-1], axis = 1)[:, ::-1]
        int_1 = Gbar[i:(N + 1), i] * s

        #B_2 = np.concatenate((B[:, 1:i], np.zeros((B.shape[0], 1))), axis = 1)
        #int_2 = np.sum(q_2(B_2, s) * DG[0:(N - i + 1), (N-end_idx):N], axis = 1)
        int_2 = np.sum(B * DG[0:(N - i + 1), (N-end_idx):N], axis = 1)
        
        integral = int_1 + int_2
        return integral


    
    
def pmfft(pgf, M=5000, immigration = None, parallel = False, n_cores = None, cutoff = None, 
         count = 'prev'):
    assert(count == 'prev' or count == 'ci'), "Count needs to be prevalence or cumulative incidence"
    
    if immigration is None:
        if parallel:
            pgf = Parallel(n_jobs=n_cores)(delayed(pgf)(np.exp(2.0j*np.pi*m/M), 
                                                  count = count) for m in range(M))
        else:
            pgf = np.array(list(map(lambda m: pgf(np.exp(2.0j*np.pi*m/M), 
                                                  count = count), range(M))))
    else:
        pgf = Parallel(n_jobs=n_cores)(delayed(pgf)(np.exp(2.0j*np.pi*m/M), 
                                                  count = count, 
                                                  immigration = immigration) for m in range(M))
        
        #pgf = np.array(list(map(lambda m: pgf(np.exp(2.0j*np.pi*m/M), 
        #                               count = count, cutoff = cutoff,
        #                               immigration),range(M))))
        
    fft = np.real(np.fft.fft(pgf, axis=0))
    pmf = fft * (fft >= 0)
    return pmf / (np.sum(pmf, axis=0)+np.finfo(float).eps)

    
def mean_pmf(pmf):
    M = np.shape(pmf)[0]
    loc = np.linspace(0, M-1, M)
    mn = np.sum(pmf.T * loc, axis = 1)
    return mn


def FirstPassageTime(pmf, Zstar, time, pdf = True):

    Zstar = int(Zstar)
    FPT_cdf = np.zeros_like(time)
    nsteps = len(time)
    time_step = time[1] - time[0]
    for i in range(nsteps):
        FPT_cdf[i] = (1-(np.cumsum(pmf[:Zstar, i]))[-1])/(1-(np.cumsum(pmf[0, i])))
    
    if pdf:
        FPT_pdf = np.gradient(FPT_cdf, time_step)
        return FPT_pdf
    else:
        return FPT_cdf
    
def FirstPassageTimeImmigration(pmf, Zstar, time, pdf = True):

    Zstar = int(Zstar)
    FPT_cdf = np.zeros_like(time)
    nsteps = len(time)
    time_step = time[1] - time[0]
    for i in range(nsteps):
        FPT_cdf[i] = (1-(np.cumsum(pmf[:Zstar, i]))[-1])
    
    if pdf:
        FPT_pdf = np.gradient(FPT_cdf, time_step)
        return FPT_pdf
    else:
        return FPT_cdf