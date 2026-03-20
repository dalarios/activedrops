# Complete the analysis: Create final motor cluster visualization
# Show NMF1 vs NMF2 with silhouette scores overlaid on the scatter plot
print("Creating final motor cluster analysis...")

from sklearn.metrics import silhouette_score, silhouette_samples
import itertools

n_components = nmf_results['n_components']
component_names = [f'Component {i+1}' for i in range(n_components)]
cluster_names = [f'Cluster {i+1}' for i in range(n_components)]
base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cluster_colors = list(itertools.islice(itertools.cycle(base_colors), n_components))

H_matrix = nmf_results['H'].T  # (conditions × components)
cluster_labels = dominant_components

# Calculate silhouette scores for existing NMF clusters (no re-clustering)
silhouette_avg = silhouette_score(H_matrix, cluster_labels)
silhouette_values = silhouette_samples(H_matrix, cluster_labels)

# Create figure with NMF scatter plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot NMF Component 1 vs Component 2 with silhouette scores as point sizes
# Use silhouette scores to determine point size (larger = better silhouette)
# Normalize silhouette scores to reasonable point sizes (50-300)
silhouette_normalized = (silhouette_values + 1) / 2  # Scale from [-1, 1] to [0, 1]
point_sizes = 50 + silhouette_normalized * 250  # Scale to [50, 300]

# Plot points colored by cluster, sized by silhouette score
for i in range(n_components):
    component_mask = cluster_labels == i
    cluster_silhouettes = silhouette_values[component_mask]
    cluster_sizes = point_sizes[component_mask]
    
    # Plot scatter with cluster color and silhouette-based size
    ax.scatter(H_matrix[component_mask, 0], H_matrix[component_mask, 1], 
               c=cluster_colors[i], s=cluster_sizes, 
               label=f'{cluster_names[i]}', 
               alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Annotate with condition names, colored by silhouette score
    for j, (x, y) in enumerate(zip(H_matrix[component_mask, 0], H_matrix[component_mask, 1])):
        condition_idx = np.where(component_mask)[0][j]
        motor_name = condition_summary.iloc[condition_idx]['Condition']
        silhouette_val = cluster_silhouettes[j]
        
        # Color annotation based on silhouette score
        if silhouette_val > 0.5:
            ann_color = 'darkgreen'
        elif silhouette_val > 0:
            ann_color = 'darkorange'
        else:
            ann_color = 'darkred'
            
        ax.annotate(motor_name, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.9, color=ann_color, fontweight='bold')

ax.set_title(f'Motor Clusters: NMF Component 1 vs Component 2\n(Silhouette Score: {silhouette_avg:.3f} | Point size ∝ silhouette score)', 
              fontsize=16, fontweight='bold')
ax.set_xlabel(f'NMF Component 1 Weight', fontsize=12)
ax.set_ylabel(f'NMF Component 2 Weight', fontsize=12)
ax.legend(title='Clusters', fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

# Add text box with silhouette interpretation
if silhouette_avg > 0.7:
    quality = "✅ EXCELLENT - Strong, well-separated clusters"
    quality_color = 'green'
elif silhouette_avg > 0.5:
    quality = "✅ GOOD - Reasonable structure"
    quality_color = 'darkgreen'
elif silhouette_avg > 0.25:
    quality = "⚠️  FAIR - Weak but potentially meaningful structure"
    quality_color = 'orange'
else:
    quality = "❌ POOR - No substantial structure"
    quality_color = 'red'

textstr = f'Overall Silhouette: {silhouette_avg:.3f}\n{quality}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor=quality_color, linewidth=2)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.show()

print(f"Final behavioral cluster analysis complete!")
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Quality: {quality}")



