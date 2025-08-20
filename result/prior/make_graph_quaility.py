import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

methods = ['WS', 'Cos', 'WJ', 'SCAN', 'TFP']
ari_values = [0.3581, 0.0997, 0.3740, 0.0066, 0.0087]

fig, ax = plt.subplots(figsize=(6, 4))
# Reduced bar width
bars = ax.bar(methods, ari_values, width=0.5, color='white', edgecolor='chocolate')

for bar in bars:
    bar.set_hatch('///')
    bar.set_linewidth(1)

ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)

ax.set_xlabel('Similarity Functions', fontsize=18)
ax.set_ylabel('Modularity', fontsize=18)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(0, 0.5)

patch = mpatches.Patch(facecolor='white', edgecolor='chocolate', hatch='///', label='Mod')
ax.legend(handles=[patch], loc='upper right', fontsize=16)

plt.tight_layout()
plt.savefig('les_miserable_mod_bar.pdf')
plt.show()