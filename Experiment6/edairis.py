import matplotlib.pyplot as plt
import numpy as np
from data import X, y, data, target
from itertools import combinations

feature_names = data.columns.tolist()
species_names = np.unique(y)

feature_pairs = list(combinations(range(len(feature_names)), 2))

n_plots = len(feature_pairs)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if n_plots > 1 else [axes]

colors = ['red', 'blue', 'green']
species_labels = species_names.flatten()

for idx, (i, j) in enumerate(feature_pairs):
    ax = axes[idx]
    
    for species_idx, species in enumerate(species_labels):
        mask = (y == species).flatten()
        ax.scatter(X[mask, i], X[mask, j], 
                  c=colors[species_idx], 
                  label=species, 
                  alpha=0.6, 
                  edgecolors='k', 
                  s=50)
    
    ax.set_xlabel(feature_names[i], fontsize=10)
    ax.set_ylabel(feature_names[j], fontsize=10)
    ax.set_title(f'{feature_names[i]} vs {feature_names[j]}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

for idx in range(n_plots, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()
