import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def plot_cor_mat(matrix):
    ax = sns.heatmap(matrix, linewidth=0.5)
    plt.title("Correlation Matrix")
    plt.show()

def graphic(x, y, title, label=None):
    plt.plot(x, y, label=label)
    plt.title(title)
    plt.show()
    
def production_graphic(x, y, label=None):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Energy production in 2020 (MW)")
    
    for i in range(9):
        if i<5:
            ax1.plot(x[i], y[i], label=label[i])
        else:
            ax2.plot(x[i], y[i], label=label[i])
            
    ax1.set_yticks([0.0, 0.5 * 1e6, 1 * 1e6, 1.5 * 1e6, 2 * 1e6, 2.5 * 1e6])
    ax2.set_yticks([0.0, 0.5 * 1e6, 1 * 1e6, 1.5 * 1e6, 2 * 1e6, 2.5 * 1e6])
    ax1.legend()
    ax2.legend()
    plt.show()
    
def graphics_part1_2(matrix, x1, x2, y1, y2, label1=None, label2=None):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle("Graphics part 1.2")
    
    sns.heatmap(matrix, ax=ax1, linewidth=0.5)
    ax1.set_title("Correlation Matrix")
    
    ax2.plot(x1, y1, label=label1)
    ax2.set_title("Energy Consumption per day in France in 2020 (MW)")
    
    for i in range(9):
        if i<5:
            ax3.plot(x2[i], y2[i], label=label2[i])
        else:
            ax4.plot(x2[i], y2[i], label=label2[i])
    
    ax3.set_title("Energy production in France in 2020 (MW)")
    ax4.set_title("Energy production in France in 2020 (MW)")
    ax3.set_yticks([0.0, 0.5 * 1e6, 1 * 1e6, 1.5 * 1e6, 2 * 1e6, 2.5 * 1e6])
    ax4.set_yticks([0.0, 0.5 * 1e6, 1 * 1e6, 1.5 * 1e6, 2 * 1e6, 2.5 * 1e6])
    ax3.legend()
    ax4.legend()
    plt.show()
    
def histogram(x, title=None):
    counts, bins = np.histogram(x)
    plt.stairs(counts, bins)
    plt.hist(bins[:-1], bins, weights=counts, color="blue")
    plt.title(title)
    plt.show()