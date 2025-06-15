import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


base_dir = "./results/"
methods = ["SEA_FFA", "SEA_FF", "SEA_RRA", "SEA_RR", "AdGDA_RR", "SEA_US", "AdGDA_US"]
plt_data = []
for method in methods :
    results = np.load(base_dir + "cifar_" + method + "_results.npy")
    plt_data.append(results) 

epochs = 200
print_skip = 10 if (epochs > 100) else 5
fig, ax = plt.subplots()


for i in range(len(methods)):
    data = np.sort(plt_data[i], axis=0) [1:-1, :]
    med = np.median(data, axis=0, keepdims=False)
    avg = np.mean(data, axis=0, keepdims=False) 
    std = np.std(data, axis=0, keepdims=False)
    
    iters = print_skip * np.arange(len(med))
    ax.plot(iters, med, label=methods[i].replace("_", "-").replace("SEA", "SEAd"))
    ax.fill_between(iters, avg-2*std, avg+2*std, alpha=0.25)

ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.set_ylim(0.40, 0.80)
plt.grid(True)
plt.legend()
plt.title("Fair Classification on CIFAR10")
ax.set_ylabel("test accuracy")
ax.set_xlabel("epochs")

plt.savefig(base_dir+"acc_plot.svg", format='svg')
plt.show()