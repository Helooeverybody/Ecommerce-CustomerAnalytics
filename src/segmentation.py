import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import numpy as np 
from pathlib import Path
import logging
from pathlib import Path
import sys
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample
from sklearn.metrics import silhouette_score, adjusted_rand_score
PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = PROJECT_ROOT / "data" / "processed"

def compare_model(X,k_range=range(2, 11),scaler=None,random_state=42):
    if scaler is not None:
        X_use = scaler.fit_transform(X)
    else:
        X_use = X
    results = []
    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=random_state
        )
        labels_km = kmeans.fit_predict(X_use)

        results.append({
            "model": "KMeans",
            "k": k,
            "silhouette": silhouette_score(X_use, labels_km)
        })
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state
        )
        labels_gmm = gmm.fit_predict(X_use)
        results.append({
            "model": "GMM",
            "k": k,
            "silhouette": silhouette_score(X_use, labels_gmm)
        })
    results_df = pd.DataFrame(results)
    pivot_df = results_df.pivot(
        index="k",
        columns="model",
        values="silhouette"
    )
    print(pivot_df)
    return results_df, pivot_df
def segmentation_(X,rfm_df,model_type="gmm", n_clusters=4,covariance_type='full',random_state=42,tsne_perplexity=25,tsne_alpha=0.7):
    if model_type.lower() == "gmm":
        model = GaussianMixture(
            n_components=int(n_clusters),
            covariance_type=covariance_type,
            random_state=random_state
        )
        labels_1 = model.fit_predict(X)
    elif model_type.lower() == "kmeans":
        model = KMeans(
            n_clusters=int(n_clusters),
            n_init=10,
            random_state=random_state
        )
        labels_1 = model.fit_predict(X)
    # 2. Stability (Bootstrap + ARI)
    X_boot, idx = resample(
        X,
        np.arange(len(X)),
        replace=True,
        random_state=random_state
    )
    labels_2 = model.fit_predict(X_boot)
    ari = adjusted_rand_score(labels_1[idx], labels_2)
    silhouette = silhouette_score(X, labels_1)

    # Assign segment
    rfm_df = rfm_df.copy()
    rfm_df[f'segment_{model_type}'] = labels_1
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate='auto',
        init='pca',
        random_state=random_state
    )
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=rfm_df[f'segment_{model_type}'],
        s=20,
        alpha=tsne_alpha
    )
    plt.colorbar(label="Cluster")
    plt.title("Customer Segmentation (GMM + t-SNE)", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
    metrics = {
        "silhouette": silhouette,
        "stability_ARI": ari
    }
    print(f"Silhouette Score : {silhouette:.4f}")
    print(f"Stability (ARI)  : {ari:.4f}")
    return rfm_df, metrics, model
def build_segment_profile(
    df,
    segment_col,
    customer_col='visitorid',
    recency_col='recency_days',
    frequency_col='frequency',
    monetary_avg_col='monetary_avg',
    monetary_sum_col='monetary_sum',
    purchase_rate_col='purchase_rate',
    tenure_col='tenure_days',
    sort_by='total_revenue',
    ascending=False
):
    profile = (
        df
        .groupby(segment_col)
        .agg(
            customers=(customer_col, 'count'),
            avg_recency=(recency_col, 'mean'),
            avg_frequency=(frequency_col, 'mean'),
            avg_monetary=(monetary_avg_col, 'mean'),
            avg_purchase_rate=(purchase_rate_col, 'mean'),
            avg_tenure=(tenure_col, 'mean'),
            total_revenue=(monetary_sum_col, 'sum')
        )
    )
    profile['revenue_share'] = (
        profile['total_revenue'] / profile['total_revenue'].sum()
    )

    quantile_70 = profile['avg_monetary'].quantile(0.7)
    def business_label(row):
        if row.avg_frequency == 1:
            return 'One-time Buyers'
        if (
            row.avg_recency < 30
            and row.avg_frequency > 5
            and row.avg_monetary > quantile_70
        ):
            return 'Champions'
        if row.avg_recency < 60 and row.avg_frequency > 4:
            return 'Loyal Customers'
        if row.avg_recency > 65 and row.avg_frequency <= 2:
            return 'At Risk'
        return 'Potential Loyalists'
    profile = profile.sort_values(sort_by, ascending=ascending)
    profile['business_label'] = profile.apply(business_label, axis=1)
    return profile
def plot_segment_analysis(
    profile_df,
    revenue_col='total_revenue',
    revenue_share_col='revenue_share',
    customer_col='customers',
    radar_metrics=None,
    normalize=True,
    figsize_bar=(8, 5),
    figsize_radar=(6, 6)
):

    profile_sorted = profile_df.sort_values(revenue_col, ascending=False)

    plt.figure(figsize=figsize_bar)
    plt.bar(
        profile_sorted.index.astype(str),
        profile_sorted[revenue_share_col]
    )
    plt.title("Revenue Contribution by Customer Segment")
    plt.xlabel("Segment")
    plt.ylabel("Revenue Share")
    plt.show()

    plt.figure(figsize=figsize_bar)
    plt.bar(
        profile_sorted.index.astype(str),
        profile_sorted[customer_col]
    )
    plt.title("Number of Customers per Segment")
    plt.xlabel("Segment")
    plt.ylabel("Customers")
    plt.show()
    
    if radar_metrics is not None:
        radar_data = profile_df[radar_metrics].copy()

        if normalize:
            radar_data = (
                radar_data - radar_data.min()
            ) / (radar_data.max() - radar_data.min())

        angles = np.linspace(
            0, 2 * np.pi, len(radar_metrics), endpoint=False
        ).tolist()
        angles += angles[:1]

        plt.figure(figsize=figsize_radar)

        for idx, row in radar_data.iterrows():
            values = row.tolist()
            values += values[:1]
            plt.plot(angles, values, label=f"Segment {idx}")
            plt.fill(angles, values, alpha=0.1)

        plt.xticks(angles[:-1], radar_metrics)
        plt.title("Customer Segment Profiles (Normalized)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.show()
def plot_segment_scatter(
    df,
    segment_col,
    recency_col='recency_days',
    frequency_col='frequency',
    monetary_col='monetary_avg',
    invert_recency=True,
    alpha=0.6,
    figsize=(7, 5)
):

    plt.figure(figsize=figsize)
    plt.scatter(
        df[recency_col],
        df[frequency_col],
        c=df[segment_col],
        alpha=alpha
    )

    if invert_recency:
        plt.gca().invert_xaxis()

    plt.xlabel("Recency (days)")
    plt.ylabel("Frequency")
    plt.title(f'Recency vs Frequency by {segment_col} ')
    plt.colorbar(label="Segment")
    plt.show()

    plt.figure(figsize=figsize)
    plt.scatter(
        df[frequency_col],
        df[monetary_col],
        c=df[segment_col],
        alpha=alpha
    )
    plt.xlabel("Frequency")
    plt.ylabel("Average Monetary Value")
    plt.title(f'Frequency vs Monetary by {segment_col}')
    plt.colorbar(label="Segment")
    plt.show()
