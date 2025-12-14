#!/usr/bin/env python3
"""
Create additional summary visualizations for task independence analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
base_dir = Path("/home/mjbommar/src/shelf-benchmark/docs/paper/issues/06_task_independence")

# Load correlation matrix
corr_df = pd.read_csv(base_dir / "task_correlations_pearson.csv", index_col=0)

# Get upper triangle (excluding diagonal)
mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
upper_tri = corr_df.where(mask).stack().values

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Correlation distribution histogram
ax1 = axes[0, 0]
ax1.hist(upper_tri, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(np.mean(upper_tri), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(upper_tri):.3f}')
ax1.axvline(np.median(upper_tri), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(upper_tri):.3f}')
ax1.set_xlabel('Correlation Coefficient (r)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Task-Task Correlations', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# 2. Box plot by correlation sign
ax2 = axes[0, 1]
positive_corr = upper_tri[upper_tri > 0]
negative_corr = upper_tri[upper_tri < 0]
near_zero = upper_tri[(upper_tri >= -0.2) & (upper_tri <= 0.2)]

box_data = [negative_corr, near_zero, positive_corr]
bp = ax2.boxplot(box_data, labels=['Negative\n(r < 0)', 'Near Zero\n(-0.2 ≤ r ≤ 0.2)', 'Positive\n(r > 0)'],
                 patch_artist=True, widths=0.6)

colors = ['lightcoral', 'lightyellow', 'lightblue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Correlation Coefficient (r)', fontsize=12)
ax2.set_title('Correlation Distribution by Sign', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add counts
ax2.text(1, -0.9, f'n={len(negative_corr)}', ha='center', fontsize=10)
ax2.text(2, -0.9, f'n={len(near_zero)}', ha='center', fontsize=10)
ax2.text(3, 0.9, f'n={len(positive_corr)}', ha='center', fontsize=10)

# 3. Load cross-type correlations
cross_type_df = pd.read_csv(base_dir / "cross_type_correlations.csv")

ax3 = axes[1, 0]
# Create bar plot for within vs cross
within_data = cross_type_df[cross_type_df['Relationship'] == 'Within']
cross_data = cross_type_df[cross_type_df['Relationship'] == 'Cross']

x_pos = np.arange(len(within_data))
width = 0.35

# Create labels for within-type
within_labels = [f"{row['Type 1']}" for _, row in within_data.iterrows()]

bars1 = ax3.bar(x_pos - width/2, within_data['Mean r'], width,
                label='Within-type', color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax3.bar(x_pos + width/2,
                [cross_data[cross_data['Type 1'] == t]['Mean r'].mean()
                 for t in within_data['Type 1']],
                width, label='Cross-type (mean)', color='coral', alpha=0.8, edgecolor='black')

ax3.set_xlabel('Task Type', fontsize=12)
ax3.set_ylabel('Mean Correlation (r)', fontsize=12)
ax3.set_title('Within-Type vs Cross-Type Correlations', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(within_labels, rotation=45, ha='right')
ax3.legend(fontsize=10)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.grid(axis='y', alpha=0.3)

# 4. Summary statistics table
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary table
summary_stats = [
    ['Statistic', 'Value'],
    ['', ''],
    ['Mean correlation', f'{np.mean(upper_tri):.3f}'],
    ['Median correlation', f'{np.median(upper_tri):.3f}'],
    ['Std deviation', f'{np.std(upper_tri):.3f}'],
    ['', ''],
    ['Min correlation', f'{np.min(upper_tri):.3f}'],
    ['Max correlation', f'{np.max(upper_tri):.3f}'],
    ['Range', f'{np.max(upper_tri) - np.min(upper_tri):.3f}'],
    ['', ''],
    ['Q1 (25th percentile)', f'{np.percentile(upper_tri, 25):.3f}'],
    ['Q3 (75th percentile)', f'{np.percentile(upper_tri, 75):.3f}'],
    ['', ''],
    ['Negative correlations', f'{np.sum(upper_tri < 0)} ({100*np.sum(upper_tri < 0)/len(upper_tri):.1f}%)'],
    ['Near-zero (-0.2 to 0.2)', f'{np.sum(np.abs(upper_tri) <= 0.2)} ({100*np.sum(np.abs(upper_tri) <= 0.2)/len(upper_tri):.1f}%)'],
    ['Positive correlations', f'{np.sum(upper_tri > 0)} ({100*np.sum(upper_tri > 0)/len(upper_tri):.1f}%)'],
    ['', ''],
    ['High correlations (r > 0.8)', f'{np.sum(upper_tri > 0.8)}'],
    ['Strong negative (r < -0.7)', f'{np.sum(upper_tri < -0.7)}'],
]

table = ax4.table(cellText=summary_stats, cellLoc='left', loc='center',
                  colWidths=[0.6, 0.4], bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style section dividers
for row in [1, 5, 9, 12, 16]:
    for col in range(2):
        table[(row, col)].set_facecolor('#f0f0f0')

ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(base_dir / "correlation_summary.png", dpi=300, bbox_inches='tight')
print(f"Saved summary visualization to {base_dir / 'correlation_summary.png'}")

# Create a second figure for champion analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Load champion data
champions_df = pd.read_csv(base_dir / "task_champions.csv")

# Champion frequency
ax_champ = axes2[0]
champion_counts = champions_df['Champion'].value_counts()
bars = ax_champ.barh(range(len(champion_counts)), champion_counts.values, color='steelblue', edgecolor='black')
ax_champ.set_yticks(range(len(champion_counts)))
ax_champ.set_yticklabels(champion_counts.index)
ax_champ.set_xlabel('Number of Task Wins', fontsize=12)
ax_champ.set_ylabel('Model', fontsize=12)
ax_champ.set_title('Task Champion Distribution (16 tasks)', fontsize=14, fontweight='bold')
ax_champ.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(champion_counts.values):
    ax_champ.text(v + 0.1, i, str(v), va='center', fontsize=10)

# Performance range by task
ax_range = axes2[1]
task_ranges = champions_df.sort_values('Range', ascending=False)
y_pos = range(len(task_ranges))

bars = ax_range.barh(y_pos, task_ranges['Range'], color='coral', edgecolor='black', alpha=0.8)
ax_range.set_yticks(y_pos)
ax_range.set_yticklabels(task_ranges['Task'], fontsize=8)
ax_range.set_xlabel('Score Range (Best - Worst)', fontsize=12)
ax_range.set_ylabel('Task', fontsize=12)
ax_range.set_title('Task Difficulty Spread', fontsize=14, fontweight='bold')
ax_range.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(base_dir / "champion_analysis.png", dpi=300, bbox_inches='tight')
print(f"Saved champion analysis to {base_dir / 'champion_analysis.png'}")

print("\nSummary statistics:")
print(f"Total task pairs: {len(upper_tri)}")
print(f"Mean correlation: {np.mean(upper_tri):.3f}")
print(f"Median correlation: {np.median(upper_tri):.3f}")
print(f"Negative correlations: {np.sum(upper_tri < 0)} ({100*np.sum(upper_tri < 0)/len(upper_tri):.1f}%)")
print(f"Near-zero (-0.2 to 0.2): {np.sum(np.abs(upper_tri) <= 0.2)} ({100*np.sum(np.abs(upper_tri) <= 0.2)/len(upper_tri):.1f}%)")
print(f"Unique champions: {champions_df['Champion'].nunique()}")
