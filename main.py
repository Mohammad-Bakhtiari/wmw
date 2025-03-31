import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def manwhitney_test(df, ds_map, alpha=0.05):
    metrics = ['NMI', 'GC', 'ILF1', 'ARI', 'EBM', 'KNN Acc', 'ASW_B', 'ASW_C']
    df.Dataset = df.Dataset.apply(lambda x: ds_map[x])
    datasets = df['Dataset'].unique()

    avg_diff = {}
    p_values = {}
    for dataset in datasets:
        p_values[dataset] = {}
        avg_diff[dataset] = {}
        df_dataset = df[df['Dataset'] == dataset]
        for metric in metrics:
            scgen_data = df_dataset[df_dataset['Approach'] == 'scGen'][metric].values
            fedscgen_data = df_dataset[df_dataset['Approach'] == 'FedscGen-SMPC'][metric].values
            if len(scgen_data) > 0 and len(fedscgen_data) > 0:
                stat, p = mannwhitneyu(scgen_data, fedscgen_data, alternative='two-sided')
                p_values[dataset][metric] = p
                avg_diff[dataset][metric] = np.mean(fedscgen_data) - np.mean(scgen_data)
            else:
                raise ValueError(f"Missing data for {dataset} - {metric}. Ensure both approaches have data.")

    flat_p_values = [p_values[d][m] for d in datasets for m in metrics if not np.isnan(p_values[d][m])]
    rejected, adj_p_values, _, _ = multipletests(flat_p_values, alpha=alpha, method='fdr_bh')
    adj_p_values = iter(adj_p_values)
    adj_p_dict = {}
    for dataset in datasets:
        adj_p_dict[dataset] = {}
        for metric in metrics:
            if not np.isnan(p_values[dataset][metric]):
                adj_p_dict[dataset][metric] = next(adj_p_values)
            else:
                raise ValueError(f"Missing data for {dataset} - {metric}")


    p_df = pd.DataFrame(adj_p_dict).T
    p_df.index.name = 'Dataset'
    p_df.columns.name = 'Metric'
    avg_df = pd.DataFrame(avg_diff).T
    avg_df.index.name = 'Dataset'
    avg_df.columns.name = 'Metric'
    wilcoxon_pvalue_heatmap(avg_df, p_df, alpha, "wilcoxon_pvalue_heatmap.png")

    # Report significant results
    significant = (p_df < alpha).sum().sum()
    print(f"Number of significant metric-dataset pairs: {significant}")
    for dataset in datasets:
        for metric in metrics:
            adj_p = adj_p_dict[dataset][metric]
            if not np.isnan(adj_p) and adj_p < alpha:
                print(f"{dataset} - {metric}: Adjusted p-value = {adj_p:.4f}")


def wilcoxon_pvalue_heatmap(avg_df, p_df, alpha, file_path):
    datasets = avg_df.index
    metrics = avg_df.columns
    annotations = avg_df.copy().astype(str)

    for i in range(len(datasets)):
        for j in range(len(metrics)):
            if not pd.isna(avg_df.iloc[i, j]):
                annotations.iloc[i, j] = f"{avg_df.iloc[i, j]:.2f}"
            else:
                raise ValueError("NaN value found in avg_df")

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(avg_df, annot=annotations, cmap='RdBu', center=0, ax=ax, cbar=False,
                annot_kws={"size": 32}, square=True, fmt="", vmin=-1, vmax=1,)

    ax.set_yticklabels(datasets, fontsize=30, rotation=0, ha='right')
    ax.set_xticklabels(metrics, fontsize=30, rotation=45, ha='right')

    cbar_ax = fig.add_axes([0.81, 0.25, 0.02, 0.5])
    norm = colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    plt.colorbar(sm, cax=cbar_ax).ax.tick_params(labelsize=24)

    for i in range(len(datasets)):
        for j in range(len(metrics)):
            p = p_df.iloc[i, j]
            stars = ""
            if not pd.isna(p):
                if p < alpha:
                    stars = "*"
            if stars:
                ax.text(j + 0.5, i + 0.3, stars, fontsize=28, ha='center', va='bottom', color='black',
                        fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(file_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    ds_map = {"HumanDendriticCells": "HDC",
                "MouseCellAtlas": "MCA",
                "HumanPancreas": "HP",
                "PBMC": "PBMC",
                "CellLine": "CL",
                "MouseRetina": "MR",
                "MouseBrain": "MB",
                "MouseHematopoieticStemProgenitorCells": "MHSPC"}

    df = pd.read_csv("smpc_wilcoxon.csv")
    manwhitney_test(df, ds_map, 0.01)