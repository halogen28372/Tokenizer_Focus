import matplotlib.pyplot as plt
import numpy as np

# Data from our runs
runs = ['Initial\n(Epoch 1)', 'New Model', 'Latest', 'Now']
s2_accuracies = [4.9, 42.0, 65.4, 69.3]
s1_accuracies = [4.9, 23.8, 40.5, 50.5]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(runs))
width = 0.35

bars1 = ax.bar(x - width/2, s1_accuracies, width, label='S1 (No TTT)', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, s2_accuracies, width, label='S2 (50 steps TTT)', color='#e74c3c', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model Checkpoint', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('S1 vs S2 Performance Progression Over Training', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 80])

plt.tight_layout()
plt.savefig('visualizations/s2_progression.png', dpi=150, bbox_inches='tight')
print('Saved progression chart to visualizations/s2_progression.png')

# Print summary table
print('\n' + '='*60)
print('S2 Performance Progression Summary')
print('='*60)
header = f"{'Run':<20} {'S1 Acc':<12} {'S2 Acc':<12} {'Improvement':<15}"
print(header)
print('-'*60)
for i, run in enumerate(runs):
    improvement = s2_accuracies[i] - s1_accuracies[i]
    row = f"{run:<20} {s1_accuracies[i]:>6.1f}%    {s2_accuracies[i]:>6.1f}%    {improvement:>+6.1f}%"
    print(row)
print('='*60)
improvement_first = s2_accuracies[1] - s1_accuracies[1]
improvement_last = s2_accuracies[-1] - s1_accuracies[-1]
print(f'\nS2 Improvement Trend: {improvement_first:.1f}% → {improvement_last:.1f}%')
s2_gain = s2_accuracies[-1] - s2_accuracies[0]
print(f'S2 Absolute Performance: {s2_accuracies[0]:.1f}% → {s2_accuracies[-1]:.1f}% (+{s2_gain:.1f}%)')

