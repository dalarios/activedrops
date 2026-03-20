# Code to replace in the final visualization cell (cell 16)
# This shows only NMF1 vs NMF2 and includes silhouette score

# Complete the analysis: Create final motor cluster visualization
# Show only NMF1 vs NMF2 with silhouette score
print("Creating final motor cluster analysis...")

from sklearn.metrics import silhouette_score
import itertools

n_components = nmf_results['n_components']
component_names = [f'Component {i+1}' for i in range(n_components)]
cluster_names = [f'Cluster {i+1}' for i in range(n_components)]
base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cluster_colors = list(itertools.islice(itertools.cycle(base_colors), n_components))

H_matrix = nmf_results['H'].T  # (conditions × components)
cluster_labels = dominant_components

# Calculate silhouette score for existing NMF clusters (no re-clustering)
silhouette_avg = silhouette_score(H_matrix, cluster_labels)

# Create figure with 2 subplots: NMF1 vs NMF2 scatter and silhouette info
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: NMF Component 1 vs Component 2 (only first two components)
ax1.set_title(f'Motor Clusters: NMF Component 1 vs Component 2\n(Silhouette Score: {silhouette_avg:.3f})', 
              fontsize=16, fontweight='bold')

for i in range(n_components):
    component_mask = cluster_labels == i
    ax1.scatter(H_matrix[component_mask, 0], H_matrix[component_mask, 1], 
               c=cluster_colors[i], label=f'{cluster_names[i]}', 
               alpha=0.8, s=100, edgecolors='black', linewidth=0.5)
    
    # Annotate with condition names
    for j, (x, y) in enumerate(zip(H_matrix[component_mask, 0], H_matrix[component_mask, 1])):
        condition_idx = np.where(component_mask)[0][j]
        motor_name = condition_summary.iloc[condition_idx]['Condition']
        ax1.annotate(motor_name, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8, color=cluster_colors[i], fontweight='bold')

ax1.set_xlabel(f'NMF Component 1 Weight', fontsize=12)
ax1.set_ylabel(f'NMF Component 2 Weight', fontsize=12)
ax1.legend(title='Clusters', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Silhouette score information
ax2.axis('off')
ax2.text(0.5, 0.9, 'Cluster Quality Assessment', ha='center', va='top', 
         fontsize=18, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.75, f'Silhouette Score: {silhouette_avg:.3f}', ha='center', va='top',
         fontsize=16, fontweight='bold', transform=ax2.transAxes,
         color='green' if silhouette_avg > 0.5 else 'orange' if silhouette_avg > 0.25 else 'red')

# Interpretation
if silhouette_avg > 0.7:
    quality = "✅ EXCELLENT - Strong, well-separated clusters"
elif silhouette_avg > 0.5:
    quality = "✅ GOOD - Reasonable structure"
elif silhouette_avg > 0.25:
    quality = "⚠️  FAIR - Weak but potentially meaningful structure"
else:
    quality = "❌ POOR - No substantial structure"

ax2.text(0.5, 0.6, quality, ha='center', va='top',
         fontsize=14, transform=ax2.transAxes)

# Cluster summary
ax2.text(0.5, 0.45, 'Cluster Summary:', ha='center', va='top',
         fontsize=14, fontweight='bold', transform=ax2.transAxes)

cluster_summary_text = []
for i in range(n_components):
    cluster_mask = cluster_labels == i
    n_samples = cluster_mask.sum()
    cluster_summary_text.append(f'{cluster_names[i]}: {n_samples} samples')

ax2.text(0.5, 0.3, '\n'.join(cluster_summary_text), ha='center', va='top',
         fontsize=12, transform=ax2.transAxes, family='monospace')

ax2.text(0.5, 0.05, 'Silhouette Score Range: [-1, 1]\n• 1: Perfect clustering\n• 0: Overlapping clusters\n• -1: Incorrect clustering',
         ha='center', va='bottom', fontsize=10, transform=ax2.transAxes,
         style='italic', alpha=0.7)

plt.tight_layout()
plt.show()

print(f"Final behavioral cluster analysis complete!")
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Quality: {quality}")
