import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import gmean 
import matplotlib.pyplot as plt 
from tqdm import tqdm, trange 

rng = np.random.default_rng(seed=42)
plt.figure(figsize=(8, 6)) 

T = 100_000

K = T // 2
rec = 1000 
skip = K // rec 
 
runs = 5
decay_exp = 0.34  

FFA_grad_norm_trace = np.zeros((runs, rec+1))
FF_grad_norm_trace = np.zeros((runs, rec+1))
RRA_grad_norm_trace = np.zeros((runs, rec+1))
RRA2_grad_norm_trace = np.zeros((runs, rec+1))
RR_grad_norm_trace = np.zeros((runs, rec+1))
USA_grad_norm_trace = np.zeros((runs, rec+1))
USA2_grad_norm_trace = np.zeros((runs, rec+1))
SEG_grad_norm_trace = np.zeros((runs, rec+1)) 

init_norm_sq = np.zeros((runs, 1))

for r in trange(runs, leave=True):   
    Q_list = []
    t_list = []
 
    dx = dy = 20
    n = 40
    ncvx = np.zeros((n, dx, dx))
    nccv = np.zeros((n, dy, dy))

    # Rademacher random diagonal hessians
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
        eta_k = eta_0 / (1. + k / 10.0)**decay_exp
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
    
    ## SEG-FF 
    z = np.copy(z0)
    FF_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        tau = rng.permutation(n)
        eta_k = eta_0 / (1. + k / 10.0)**decay_exp
        z_anch = np.copy(z)
        for i in range(n):
            w = z - eta_k * F_hat(z, tau[i]) 
            z = z - eta_k * F_hat(w, tau[i])
        for i in range(n):
            w = z - eta_k * F_hat(z, tau[n-1-i])
            z = z - eta_k * F_hat(w, tau[n-1-i])

        if (k+1) % skip == 0 :
            FF_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2
    
    ## SEG-RRA
    z = np.copy(z0)
    RRA_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.permutation(n)
            eta_k = eta_0 / (1. + k / 10.0)**decay_exp
            z_anch = np.copy(z)
            for i in range(n):
                w = z - eta_k * F_hat(z, tau[i]) 
                z = z - eta_k * F_hat(w, tau[i])

            z = 0.5 * (z + z_anch)

        if (k+1) % skip == 0 :
            RRA_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

    ## SEG-RRA2
    z = np.copy(z0)
    RRA2_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.permutation(n)
            eta_k = eta_0 / (1. + k / 10.0)**decay_exp
            z_anch = np.copy(z)
            for i in range(n):
                w = z - eta_k/2 * F_hat(z, tau[i]) 
                z = z - eta_k * F_hat(w, tau[i])

            z = 0.5 * (z + z_anch)

        if (k+1) % skip == 0 :
            RRA2_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

    ## SEG-RR
    z = np.copy(z0)
    RR_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.permutation(n)
            eta_k = eta_0 / (1. + k / 10.0)**decay_exp 
            for i in range(n):
                w = z - eta_k * F_hat(z, tau[i]) 
                z = z - eta_k * F_hat(w, tau[i])
        
        if (k+1) % skip == 0 :
            RR_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q @ z - t) ** 2 
    
    ## SEG-USA
    z = np.copy(z0)
    USA_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.choice(n, n, replace=True)
            eta_k = eta_0 / (1. + k / 10.0)**decay_exp
            z_anch = np.copy(z)
            for i in range(n):
                w = z - eta_k * F_hat(z, tau[i]) 
                z = z - eta_k * F_hat(w, tau[i])
            
                z = 0.5 * (z + z_anch)

        if (k+1) % skip == 0 :
            USA_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

    ## SEG-USA2
    z = np.copy(z0)
    USA2_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.choice(n, n, replace=True)
            eta_k = eta_0 / (1. + k / 10.0)**decay_exp
            z_anch = np.copy(z)
            for i in range(n):
                w = z - eta_k/2 * F_hat(z, tau[i]) 
                z = z - eta_k * F_hat(w, tau[i])
            
                z = 0.5 * (z + z_anch)

        if (k+1) % skip == 0 :
            USA2_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

    ## SEG-US
    z = np.copy(z0)
    SEG_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.choice(n, n, replace=True)
            eta_k = eta_0 / (1. + k / 10.0)**decay_exp 
            for i in range(n):
                w = z - eta_k * F_hat(z, tau[i]) 
                z = z - eta_k * F_hat(w, tau[i]) 

        if (k+1) % skip == 0 :
            SEG_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

        
FFA_grad_norm_trace /= init_norm_sq
FF_grad_norm_trace /= init_norm_sq
RRA_grad_norm_trace /= init_norm_sq
RRA2_grad_norm_trace /= init_norm_sq
RR_grad_norm_trace /= init_norm_sq
USA_grad_norm_trace /= init_norm_sq
USA2_grad_norm_trace /= init_norm_sq
SEG_grad_norm_trace /= init_norm_sq 

FFA_mean = gmean(FFA_grad_norm_trace, axis=0)  
FF_mean = gmean(FF_grad_norm_trace, axis=0)  
RRA_mean = gmean(RRA_grad_norm_trace, axis=0)  
RRA2_mean = gmean(RRA2_grad_norm_trace, axis=0)  
RR_mean = gmean(RR_grad_norm_trace, axis=0)  
USA_mean = gmean(USA_grad_norm_trace, axis=0)  
USA2_mean = gmean(USA2_grad_norm_trace, axis=0)   
SEG_mean = gmean(SEG_grad_norm_trace, axis=0)  

xtikz = 2*skip * np.arange(0, rec+1)
plt.semilogy(xtikz, FFA_mean, alpha=0.67)
plt.semilogy(xtikz, FF_mean, alpha=0.67)
plt.semilogy(xtikz, RR_mean, alpha=0.67)
plt.semilogy(xtikz, SEG_mean, alpha=0.67)
plt.semilogy(xtikz, RRA_mean, "--", alpha=0.9)
plt.semilogy(xtikz, RRA2_mean, "--", alpha=0.9)
plt.semilogy(xtikz, USA_mean, "--", alpha=0.9)
plt.semilogy(xtikz, USA2_mean, "--", alpha=0.9)

dat = np.vstack((FFA_mean, FF_mean, RRA_mean, RRA2_mean, RR_mean,
                 USA_mean, USA2_mean, SEG_mean))
np.save("CC-50000.npy", dat)
 
plt.rcParams['text.usetex'] = True
plt.xlabel("number of passes ($t$)", fontsize=15)
plt.gca().set_ylim(top=1e+6) 
plt.xticks(fontsize=15)
plt.ylabel(r"$\dfrac{\|F z_0^t\|^2}{\|F z_0^0\|^2}$ or $\dfrac{\|F z_0^{t/2}\|^2}{\|F z_0^0\|^2}$", fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.legend(["SEG-FFA", "SEG-FF", "SEG-RR", "SEG-US",
            r"SEG-RRA, $\alpha = \beta$", r"SEG-RRA, $\alpha = \beta/2$",
            r"SEG-USA, $\alpha = \beta$", r"SEG-USA, $\alpha = \beta/2$",],
           fontsize=15, ncol=2)


plt.savefig('monotone_results.pdf', bbox_inches='tight')
plt.show()
