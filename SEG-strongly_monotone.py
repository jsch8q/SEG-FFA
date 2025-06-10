import numpy as np
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt 
from tqdm import tqdm, trange 

rng = np.random.default_rng(seed=42)
plt.figure(figsize=(8, 6))

## Change the number of passes (T) and the stepsize (eta_k) as needed. 
T = 800
eta_k = 0.001

K = T // 2
rec = 100
skip = K // rec

runs = 5 

FFA_grad_norm_trace = np.zeros((runs, rec+1))
FF_grad_norm_trace = np.zeros((runs, rec+1))
RRA_grad_norm_trace = np.zeros((runs, rec+1))
RR_grad_norm_trace = np.zeros((runs, rec+1))
USA_grad_norm_trace = np.zeros((runs, rec+1))
SEG_grad_norm_trace = np.zeros((runs, rec+1))
SGDA_grad_norm_trace = np.zeros((runs, rec+1))
SGDARR_grad_norm_trace = np.zeros((runs, rec+1))

init_norm_sq = np.zeros((runs, 1))

for r in trange(runs, leave=True): 

    Q_list = []
    t_list = [] 

    dx = dy = 20
    n = 40
    for i in range(n):
        B = rng.uniform(size=(dx, dy)) 
        Q_A, _ = np.linalg.qr(rng.standard_normal((dx, dx)))
        Q_C, _ = np.linalg.qr(rng.standard_normal((dy, dy)))
        A = Q_A @ np.diag(0.5+0.5*rng.uniform(size=dx)) @ Q_A.T
        C = Q_C @ np.diag(0.5+0.5*rng.uniform(size=dy)) @ Q_C.T
        Qi = np.block([[A, B], [B.T, -C]])
        ti = rng.standard_normal(dx+dy)
        Q_list.append(Qi)
        t_list.append(ti)

    
    Q = 1./n * sum(Q_list)
    t = 1./n * sum(t_list)
    sdl = np.concatenate((np.ones(dx), -np.ones(dy)))
    z_opt = np.linalg.lstsq(Q, t, rcond=None)[0]

    F_hat = lambda z, xi : sdl * (Q_list[xi] @ z - t_list[xi])  
    z0 = z_opt + np.ones(dx+dy) / np.linalg.norm(np.ones(dx+dy))
    init_norm_sq[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    ## SEG-FFA
    z = np.copy(z0)
    FFA_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        tau = rng.permutation(n) 
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
        z_anch = np.copy(z)
        for i in range(n):
            w = z - eta_k * F_hat(z, tau[i]) 
            z = z - eta_k * F_hat(w, tau[i])
        for i in range(n):
            w = z - eta_k * F_hat(z, tau[n-1-i])
            z = z - eta_k * F_hat(w, tau[n-1-i])

        if (k+1) % skip == 0 :
            FF_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

    ## SEG-RR
    z = np.copy(z0)
    RR_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.permutation(n)
            for i in range(n):
                w = z - eta_k * F_hat(z, tau[i]) 
                z = z - eta_k * F_hat(w, tau[i])
        
        if (k+1) % skip == 0 :
            RR_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q @ z - t) ** 2 

    ## SEG-US
    z = np.copy(z0)
    SEG_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.choice(n, n, replace=True) 
            for i in range(n):
                w = z - eta_k * F_hat(z, tau[i]) 
                z = z - eta_k * F_hat(w, tau[i]) 

        if (k+1) % skip == 0 :
            SEG_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

    ## SGDA-RR
    z = np.copy(z0)
    SGDARR_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.choice(n, n, replace=False) 
            for i in range(n): 
                z = z - eta_k * F_hat(z, tau[i]) 

        if (k+1) % skip == 0 :
            SGDARR_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t)**2

    ## SGDA-US
    z = np.copy(z0)
    SGDA_grad_norm_trace[r, 0] = np.linalg.norm(Q @ z0 - t)**2

    for k in tqdm(range(K), leave=False):
        for _ in range(2):
            tau = rng.choice(n, n, replace=True) 
            for i in range(n): 
                z = z - eta_k * F_hat(z, tau[i]) 

        if (k+1) % skip == 0 :
            SGDA_grad_norm_trace[r, (k+1)//skip] = np.linalg.norm(Q@z - t) ** 2

FFA_grad_norm_trace /= init_norm_sq
FF_grad_norm_trace /= init_norm_sq
RR_grad_norm_trace /= init_norm_sq
SEG_grad_norm_trace /= init_norm_sq
SGDARR_grad_norm_trace /= init_norm_sq
SGDA_grad_norm_trace /= init_norm_sq

FFA_mean = gmean(FFA_grad_norm_trace, axis=0) 
FF_mean = gmean(FF_grad_norm_trace, axis=0)
RR_mean = gmean(RR_grad_norm_trace, axis=0)
SEG_mean = gmean(SEG_grad_norm_trace, axis=0)
SGDARR_mean = gmean(SGDARR_grad_norm_trace, axis=0)
SGDA_mean = gmean(SGDA_grad_norm_trace, axis=0) 

xtikz = 2*skip*np.arange(0, rec+1)
plt.semilogy(xtikz, FFA_mean, alpha=0.9)
plt.semilogy(xtikz, FF_mean, alpha=0.9) 
plt.semilogy(xtikz, RR_mean, alpha=0.9) 
plt.semilogy(xtikz, SEG_mean, alpha=0.9)
plt.semilogy(xtikz, SGDARR_mean, '--', alpha=0.9)
plt.semilogy(xtikz, SGDA_mean, '--', alpha=0.9) 

plt.rcParams['text.usetex'] = True 
plt.xlabel("number of passes ($t$)", fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel(r"$\dfrac{\|F z_0^t\|^2}{\|F z_0^0\|^2}$ or $\dfrac{\|F z_0^{t/2}\|^2}{\|F z_0^0\|^2}$", fontsize=15)
plt.yticks(fontsize=15) 
plt.tight_layout() 

plt.legend(["SEG-FFA", \
            "SEG-FF", \
            "SEG-RR", \
            "SEG-US", \
            "SGDA-RR", \
            "SGDA-US",], ncols=3, fontsize=15)

plt.savefig('geom_mean_scsc.pdf', bbox_inches='tight')        
plt.show()
