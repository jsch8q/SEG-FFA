import numpy as np
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt 
from tqdm import tqdm, trange 

rng = np.random.default_rng(seed=42)
plt.figure(figsize=(8, 6))


T = 10000

K = T // 2
rec = 1000
skip = K // rec 
 
runs = 5
decay_exp = 0.34  

FFA_grad_norm_trace = np.zeros((runs, rec+1)) 
SEG_grad_norm_trace = np.zeros((runs, rec+1))
SEG2_grad_norm_trace = np.zeros((runs, rec+1))
   
init_norm_sq = np.zeros((runs, 1))

for r in trange(runs, leave=True):   

    Q_list = []
    t_list = []

    dx = dy = 20
    n = 40
    ncvx = np.zeros((n, dx, dx))
    nccv = np.zeros((n, dy, dy))

    # option 1: Rademacher random diagonal hessians
    
    for i in range(dx):
        neg = rng.choice(n, size=(n//2,), replace=False)
        ncvx[:, i, i] = 1
        for idx in neg :
            ncvx[idx, i, i] = -1
    for i in range(dx):
        neg = rng.choice(n, size=(n//2,), replace=False)
        nccv[:, i, i] = 1
        for idx in neg :
            nccv[idx, i, i] = -1

    # Construct component functions
    ncvx_mean = np.mean(ncvx, axis=0)
    nccv_mean = np.mean(nccv, axis=0)
    for i in range(n):
        ncvx[i] -= ncvx_mean
        nccv[i] -= nccv_mean
            
    for i in range(n):
        B = rng.uniform(size=(dx, dy)) 
        A = np.zeros((dx, dx)) + 2*ncvx[i]
        C = np.zeros((dy, dy)) + 2*nccv[i] 
            
        Qi = np.block([[A, B], [B.T, -C]])
        ti = rng.standard_normal(dx+dy)
        Q_list.append(Qi)
        t_list.append(ti)


    Q = 1./n * sum(Q_list)
    t = 1./n * sum(t_list)
    sdl = np.concatenate((np.ones(dx), -np.ones(dy)))
    z_opt = np.linalg.lstsq(Q, t, rcond=None)[0]

    F_hat = lambda z, xi : sdl * (Q_list[xi] @ z - t_list[xi])  
    eta_0 = min(0.01, 1.0/np.linalg.norm(Q))
    z0 = z_opt + np.ones(dx+dy)  
    
    init_norm_sq[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    ## SEG-FFA
    z = np.copy(z0)
    FFA_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        tau = rng.permutation(n)
        eta_k = eta_0 / (1. + k/10)**decay_exp
        z_anch = np.copy(z)
        for i in range(n):
            w = z - eta_k/2 * F_hat(z, tau[i]) 
            z = z - eta_k * F_hat(w, tau[i])
        for i in range(n):
            w = z - eta_k/2 * F_hat(z, tau[n-1-i])
            z = z - eta_k * F_hat(w, tau[n-1-i])
        z = 0.5 * (z_anch + z)
        
        if (k+1) % skip == 0 :
            FFA_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

    ## DSEG by Hsieh et al., exponents (0, 1)
    z = np.copy(z0)
    SEG_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2
    
    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau1 = rng.choice(n, n, replace=True)
            tau2 = rng.choice(n, n, replace=True)
             
            for i in range(n):
                tt = k*n+i
                gamma_k = 1
                eta_k = 0.1 / (tt + 19.0) 
                w = z - gamma_k * F_hat(z, tau1[i]) 
                z = z - eta_k * F_hat(w, tau2[i]) 

        if (k+1) % skip == 0 :
            SEG_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

    ## DSEG by Hsieh et al., exponents (1/3, 2/3)
    z = np.copy(z0)
    SEG2_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2
    
    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau1 = rng.choice(n, n, replace=True)
            tau2 = rng.choice(n, n, replace=True)
             
            for i in range(n):
                tt = k*n+i
                gamma_k = 0.1 / (tt + 19.0)**(1/3)
                eta_k = 0.05 / (tt + 19.0)**(2/3) 
                w = z - gamma_k * F_hat(z, tau1[i]) 
                z = z - eta_k * F_hat(w, tau2[i]) 

        if (k+1) % skip == 0 :
            SEG2_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

FFA_grad_norm_trace /= init_norm_sq
SEG_grad_norm_trace /= init_norm_sq
SEG2_grad_norm_trace /= init_norm_sq

FFA_mean = gmean(FFA_grad_norm_trace, axis=0)
SEG_mean = gmean(SEG_grad_norm_trace, axis=0)
SEG2_mean = gmean(SEG2_grad_norm_trace, axis=0)

dat = np.vstack((FFA_mean, SEG_mean, SEG2_mean))
np.save("Hsieh-50000.npy", dat)
 
xtikz = 2*skip * np.arange(0, rec+1)
plt.semilogy(xtikz, FFA_mean, alpha=0.9)
plt.semilogy(xtikz, SEG_mean, alpha=0.9)
plt.semilogy(xtikz, SEG2_mean, alpha=0.9) 

# plt.gca().set_ylim(top=1e+6)
plt.rcParams['text.usetex'] = True
plt.xlabel("number of passes ($t$)", fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel(r"$\dfrac{\|F z_0^t\|^2}{\|F z_0^0\|^2}$ or $\dfrac{\|F z_0^{t/2}\|^2}{\|F z_0^0\|^2}$", fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()

plt.legend(["SEG-FFA", \
            "DSEG, exponents (0, 1)", \
            "DSEG, exponents (1/3, 2/3)"], fontsize=15)

plt.savefig('geom_mean_hsieh.pdf', bbox_inches='tight')
        
plt.show()
