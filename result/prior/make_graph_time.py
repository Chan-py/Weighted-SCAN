import matplotlib.pyplot as plt

# Graph sizes and corresponding runtimes (max of two runs)
sizes = [1000, 2000, 5000, 10000, 20000]

runtime_ws   = [0.174, 0.469, 2.757, 3.218, 10.608]
runtime_cos  = [0.186, 0.534, 1.186, 3.964, 13.982]
runtime_wj   = [3.399, 14.795, 78.986, 333.644, 1392.53]
runtime_scan = [0.270, 1.037, 4.268, 19.950, 95.934]
runtime_tfp  = [0.101, 0.376, 1.453, 5.500, 22.008]

methods = ['WS', 'Cos', 'WJ', 'SCAN', 'WSCAN-TFP']
runtimes = [runtime_ws, runtime_cos, runtime_wj, runtime_scan, runtime_tfp]

# Distinct, well‐separated colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'x']

plt.figure(figsize=(6,4))
for method, rt, color, marker in zip(methods, runtimes, colors, markers):
    plt.plot(sizes, rt,
             marker=marker, markersize=6, linestyle='-',
             markerfacecolor='white', markeredgecolor=color,
             color=color, linewidth=1.5, label=method)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of nodes (log scale)', fontsize=12)
plt.ylabel('Runtime (s, log scale)', fontsize=12)
plt.title('Algorithm Runtime vs. Graph Size', fontsize=14)
plt.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(title='Similarity', frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig('runtime_lines_distinct_colors.pdf')
plt.show()