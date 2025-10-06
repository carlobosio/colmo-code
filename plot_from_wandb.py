import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Apply seaborn styles
sns.set_style("whitegrid")
sns.set_context("paper")

wandb_export = "wandb_export_2025-09-29T14_01_50.849-07_00.csv"
df = pd.read_csv(wandb_export)

naming = {'Cheetah': ('bumbling-donkey-174',
                      'cosmic-glitter-145'), #wo
          'Quadruped': ('eternal-jazz-113', 
                        'misunderstood-dust-152'),
          'Unitree': ('cosmic-puddle-127', 
          'olive-planet-173')}

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

title_fontsize = 20
label_fontsize = 16
legend_fontsize = 14

for ax_id, (label, (with_id, without_id)) in enumerate(naming.items()):
    vis_with = True
    vis_without = True

    for col in df.columns:
        if with_id in col and not ("__MIN" in col or "__MAX" in col):

            # label =  #else 'Without Best-Score' # not in axs[ax_id].get_legend_handles_labels()[1]  else ('Without' if not " best_score" in col and 'Without' not in axs[ax_id].get_legend_handles_labels()[1] else "")
            if " best_score" in col:
                color = 'navy'
                linewidth = 2.5
                alpha = 1.0
                zorder = 10
                linestyle = '-'
                legend_label = 'Best-Island with Optim'
            else:
                color = 'lightblue'
                linewidth = 1
                alpha = 0.8
                zorder = 2
                linestyle = '--'
                legend_label = 'Suboptimal-Island with Optim' if vis_with else ""
                vis_with = False
  

            axs[ax_id].plot(df[col], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, linestyle=linestyle, label=legend_label)
        if without_id in col and not ("__MIN" in col or "__MAX" in col):
            if " best_score" in col:
                color = 'darkred'
                linewidth = 2.5
                alpha = 1.0
                zorder = 10
                linestyle = '-'
                legend_label = 'Best-Island without Optim'

            else:
                color = 'lightcoral'
                linewidth = 1
                alpha = 0.8
                zorder = 2
                linestyle = '--'
                legend_label = 'Suboptimal-Island with Optim' if vis_without else ""
                vis_without = False

            axs[ax_id].plot(df[col], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, linestyle=linestyle,  label=legend_label)

    axs[ax_id].set_title(label, fontsize=title_fontsize)
    axs[ax_id].set_xlabel('Iteration', fontsize=label_fontsize)
    axs[ax_id].set_ylabel('Best Score', fontsize=label_fontsize)
    if ax_id == 0:
        axs[ax_id].legend(fontsize=legend_fontsize, loc='lower right') # best
    axs[ax_id].grid(True) 

# fig.suptitle('Policy Seach Runs', fontsize=title_fontsize + 2)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

fig.savefig('fig_policy_search.png', dpi=300)

