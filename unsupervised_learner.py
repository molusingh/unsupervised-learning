import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.manifold import TSNE

RANDOM_SEED = 1994540101
np.random.seed(RANDOM_SEED) # keep results consistent
smote = SMOTE(random_state=RANDOM_SEED)


"""
Splits data into training and test, handles imbalances using SMOTE, and scales using Standard scaler
"""
def process_data(x, y, ts=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts) # 70% training and 30% test
    x_train, y_train = smote.fit_resample(x_train, y_train)

    rs = StandardScaler()
    x_train = rs.fit_transform(x_train)
    x_test = rs.transform (x_test)
    x_train = pd.DataFrame(x_train, columns=x.columns)
    x_test = pd.DataFrame(x_test, columns=x.columns)

    return x_train, x_test, y_train, y_test

def searchk_kmeans(x, num_clusters=range(1,11)):
    inertias = []
    silhouettes = []
    for i in num_clusters:
        result = KMeans(n_clusters=i, init='random', n_init=10, max_iter=100, tol=1e-04, random_state=RANDOM_SEED)
        result.fit(x)
        if result.n_iter_ >= result.max_iter:
            print("warning: kmeans didn't converge")
        inertias.append(result.inertia_)
        if i > 1:
            silhouettes.append(silhouette_score(x, result.labels_))
        else:
            silhouettes.append(float('-inf'))
    silhouettes = np.array(silhouettes)
    best_n_clusters = num_clusters[silhouettes.argmax()]
    print(f'Best silhouette score: {silhouettes.max()} for number of clusters: {best_n_clusters}')
    return inertias, num_clusters, best_n_clusters

def search_components_gm(x, dims=range(1, 11)):
    bic_results = []
    for i in dims:
        result = GaussianMixture(n_components=i, random_state=RANDOM_SEED, n_init=10)
        result.fit(x)
        bic_results.append(result.bic(x))
    best_n_components = dims[np.array(bic_results).argmin()]
    return bic_results, dims, best_n_components

def plot_data(x, y, xlabel=None, ylabel=None, title=None, output=None):
    plt.plot(x, y, marker='o')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.tight_layout()

    if output is not None:
        plt.savefig(output)
        plt.close()
    else:
        plt.show()

def plot_tsne(x, labels, num_labels, perplexity=30., title=None, output=None):
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    x_2d = tsne.fit_transform(x)
    x_2d = pd.DataFrame(x_2d,columns=['tsne-1', 'tsne-2'])
    x_2d['label'] = labels
    sns.scatterplot(x="tsne-1", y="tsne-2",hue="label",palette=sns.color_palette("hls", num_labels),data=x_2d,legend="full")
    if title is not None:
        plt.title(title)
    plt.tight_layout()

    if output is not None:
        plt.savefig(output)
        plt.close()
    else:
        plt.show()

def run_kmeans(x, output=None, caption=''):
    inertias, num_clusters, best_n_clusters = searchk_kmeans(x)
    plot_config = {
        "x": num_clusters,
        "y": inertias,
        "xlabel": 'Number of clusters',
        "ylabel": 'Sum of Squared Distances',
        "title": f'{caption}-K Means: Number of Clusters vs Sum of Squared Distances',
        'output': f'{output}/{caption}-kmeans-inertias' if output is not None else None
    }
    plot_data(**plot_config)
    result = KMeans(n_clusters=best_n_clusters, init='random', n_init=10, max_iter=100, tol=1e-04, random_state=RANDOM_SEED)
    result.fit(x)
    tsne_output = f'{output}/{caption}-kmeans-tsne' if output is not None else None
    plot_tsne(x, result.labels_, best_n_clusters, title=f"{caption}-TSNE Cluster Visualization K Means with {best_n_clusters} clusters", output=tsne_output)
    x_results = x.copy()
    x_results['label'] = result.labels_
    x_results.to_csv(f'{output}/{caption}-kmeans.csv', index=False)

def run_gm(x, output=None, caption=''):
    bic_results, dims, n_comp = search_components_gm(x)
    plot_config = {
        "x": dims,
        "y": bic_results,
        "xlabel": 'BIC',
        "ylabel": 'BIC',
        "title": f'{caption}-EM: Number of Components vs BIC',
        'output': f'{output}/{caption}-em-bic' if output is not None else None
    }
    plot_data(**plot_config)
    result = GaussianMixture(n_components=n_comp, random_state=RANDOM_SEED, n_init=10)
    result.fit(x)
    labels = result.predict(x)
    tsne_output = f'{output}/{caption}-em-tsne' if output is not None else None
    plot_tsne(x, labels, n_comp, title=f"{caption}-TSNE Cluster Visualization EM  with {n_comp} components", output=tsne_output)
    x_results = x.copy()
    x_results['label'] = labels
    x_results.to_csv(f'{output}/{caption}-kmeans.csv', index=False)

def run_pca(x, output=None, caption=''):
    pca = PCA(random_state=RANDOM_SEED)
    pca.fit_transform(x)
    x_pca = pca.transform(x)
    plot_config = {
        "x": range(1, 1 + x_pca.shape[1]),
        "y": np.cumsum(pca.explained_variance_ratio_) * 100,
        "xlabel": 'Number of components',
        "ylabel": 'Explained Variance',
        "title": 'PCA: Number of Components vs Explained Variance',
        "output": f'{output}/{caption}-pca-num_components' if output is not None else None
    }
    plot_data(**plot_config)

    pca = PCA(0.90, random_state=RANDOM_SEED)
    x_pca = pca.fit_transform(x)
    num_components = pca.n_components_
    x_pca = pd.DataFrame(x_pca)
    x_pca.to_csv(f'{output}/{caption}-pca.csv', index=False)
    print(f'PCA reduction: from {x.shape[1]} to {num_components}')
    return x_pca, num_components

def run_ica(x, output=None, caption=''):
    kurt_results = []
    n_range = range(2, x.shape[1] + 1)
    for i in n_range:
        ica = FastICA(n_components=i, random_state=RANDOM_SEED)
        x_ica = ica.fit_transform(x)
        kurt_results.append(pd.DataFrame(x_ica).kurt(axis=0).abs().mean())
    plot_config = {
        "x": n_range,
        "y": kurt_results,
        "xlabel": 'Number of components',
        "ylabel": 'Kurtosis',
        "title": 'ICA: Number of Components vs Kurtosis',
        "output": f'{output}/{caption}-ica-num_components' if output is not None else None
    }
    plot_data(**plot_config)
    n = n_range[np.array(kurt_results).argmax()]
    ica = FastICA(n_components=n, random_state=RANDOM_SEED)
    x_ica = ica.fit_transform(x)
    x_ica = pd.DataFrame(x_ica)
    x_ica.to_csv(f'{output}/{caption}-ica.csv', index=False)
    print(f'ICA reduction: from {x.shape[1]} to {n}')
    return x_ica, n

def run_rp(x, threshold=0.35, output=None, caption=''):
    loss_results = []
    n_range = range(2, x.shape[1] + 1)
    for n in n_range:
        transformer = GaussianRandomProjection(random_state=RANDOM_SEED, n_components=n)
        x_rp = transformer.fit_transform(x) 
        x_recon = x_rp @ np.linalg.pinv(transformer.components_.T)
        loss_results.append(mean_squared_error(x, x_recon))

    plot_config = {
        "x": n_range,
        "y": loss_results,
        "xlabel": 'Number of components',
        "ylabel": 'Reconstruction Error',
        "title": 'ICA: Number of Components vs Reconstruction Error',
        "output": f'{output}/{caption}-rp-num_components' if output is not None else None
    }
    plot_data(**plot_config)
    for i in range(len(loss_results)):
        if loss_results[i] < threshold:
            break
    n = n_range[i]
    transformer = GaussianRandomProjection(random_state=RANDOM_SEED, n_components=n)
    x_rp = transformer.fit_transform(x)
    x_rp = pd.DataFrame(x_rp)
    x_rp.to_csv(f'{output}/{caption}-rp.csv', index=False)
    print(f'RP reduction: from {x.shape[1]} to {n}')
    return loss_results, x_rp, n

def run_experiment_1(x, dataset, output):
    print("\nRunning Experiment1:\nrunning kmeans...")
    run_kmeans(x, output=output, caption=f'{dataset}')
    print('running gaussian mixture / em...')
    run_gm(x, output=output, caption=f'{dataset}')

def run_experiment_2(x, dataset, output):
    print("\nRunning Experiment2:\nrunning pca...")
    run_pca(x, output=output, caption=dataset)
    print('running ica...')
    run_ica(x, output=output, caption=dataset)
    print('running rp...')
    run_rp(x, output=output, caption=dataset)

def run_experiment_3(dataset, output):
    x_pca = pd.read_csv(f'{output}/{dataset}-pca.csv')
    print("\nRunning Experiment3:\nrunning for pca data:\nrunning kmeans...")
    run_kmeans(x_pca, output=output, caption=f'{dataset}-pca')
    print('running gaussian mixture / em...')
    run_gm(x_pca, output=output, caption=f'{dataset}-pca')

    x_ica = pd.read_csv(f'{output}/{dataset}-ica.csv')
    print("\nrunning for ica data:\nrunning kmeans...")
    run_kmeans(x_ica, output=output, caption=f'{dataset}-ica')
    print('running gaussian mixture / em...')
    run_gm(x_ica, output=output, caption=f'{dataset}-ica')

    x_rp = pd.read_csv(f'{output}/{dataset}-rp.csv')
    print("\nrunning for rp data:\nrunning kmeans...")
    run_kmeans(x_rp, output=output, caption=f'{dataset}-rp')
    print('running gaussian mixture / em...')
    run_gm(x_rp, output=output, caption=f'{dataset}-rp')
