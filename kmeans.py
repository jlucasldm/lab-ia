from math import log
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from logging import getLogger, StreamHandler, INFO
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, rand_score, jaccard_score, fowlkes_mallows_score

def setup_argparser():
    parser = argparse.ArgumentParser(description='Treinar um modelo de decisão')
    parser.add_argument('--dataset-path', type=str, required=True, help='Caminho do dataset de input')
    parser.add_argument('--output-path', type=str, required=True, help='Diretório de output para os modelos treinados')
    parser.add_argument('--clustering-algorithm', type=str, default='kmeans', help='Algoritmo de clustering')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--target-column', type=str, default='classificacaoFinal', help='Nome da coluna alvo')
    parser.add_argument('--silent', action='store_true', help='Não exibir mensagens de log')
    parser.add_argument('--optimal-clusters', type=int, default=5, help='Número ótimo de clusters')
    parser.add_argument('--elbow-method', action='store_true', help='Usar método Elbow para calcular número ótimo de clusters')
    parser.add_argument('--silhouette-analysis', action='store_true', help='Usar Silhouette Analysis para calcular número ótimo de clusters')
    parser.add_argument('--rand-score', action='store_true', help='Usar Rand Score para calcular número ótimo de clusters')
    parser.add_argument('--jaccard-score', action='store_true', help='Usar Jaccard Score para calcular número ótimo de clusters')
    parser.add_argument('--fm-score', action='store_true', help='Usar Fowlkes-Mallows Score para calcular número ótimo de clusters')
    parser.add_argument('--save-plot', action='store_true', help='Salvar plot de clusters')
    return parser

def setup_logger(silent):
    logger = getLogger()
    logger.setLevel(INFO)
    handler = StreamHandler()
    handler.setLevel(INFO)
    logger.addHandler(handler)
    if silent:
        logger.disabled = True
    return logger

def main():
    argparser = setup_argparser()
    args = argparser.parse_args()
    logger = setup_logger(args.silent)

    logger.info(f'Carregando dataset com caminho {args.dataset_path}')
    dataset = pd.read_csv(args.dataset_path)

    features = dataset.drop(columns=[args.target_column])
    target = dataset[args.target_column]
    optimal_clusters = args.optimal_clusters

    pca = PCA(n_components=50)
    reduced_features = pca.fit_transform(features)
    #reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

    clustering_algorithm = None

    if args.clustering_algorithm == 'kmeans':
        logger.info(f'Usando K-Means para clusterização com {optimal_clusters} clusters')
        clustering_algorithm = KMeans(n_clusters=optimal_clusters, random_state=42)
        clustering_algorithm.fit(reduced_features)

        dataset['Cluster'] = clustering_algorithm.labels_

    if args.clustering_algorithm == 'agglomerative':
        logger.info(f'Usando Agglomerative Clustering para clusterização com {optimal_clusters} clusters')
        clustering_algorithm = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
        dataset['Cluster'] = clustering_algorithm.fit_predict(reduced_features)

    if args.elbow_method:
        sse = []
        logger.info('Usando método Elbow para calcular número ótimo de clusters')
        for k in range(1, 11):
            logger.info(f'Calculando SSE para {k} clusters')
            kmeans = KMeans(n_clusters=k, random_state=args.random_state)
            kmeans.fit(features)
            logger.info(f'SSE calculado: {kmeans.inertia_}')
            sse.append(kmeans.inertia_)

        plt.plot(range(1, 11), sse)
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        plt.savefig(f'{args.output_path}/elbow_method.png')

    if args.silhouette_analysis:
        silhouette_scores = []
        logger.info('Calculando Silhouette Score para diferentes números de clusters')
        for k in range(2, 11):
            logger.info(f'Calculando Silhouette Score para {k} clusters')
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            score = silhouette_score(features, kmeans.labels_)
            logger.info(f'Silhouette Score calculado: {score}')
            silhouette_scores.append(score)
        plt.plot(range(2, 11), silhouette_scores)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.savefig(f'{args.output_path}/silhouette_analysis.png')

    if args.rand_score:
        rand = rand_score(target, dataset['Cluster'])
        logger.info(f'Rand Score: {rand}')
    
    if args.jaccard_score:
        jaccard = jaccard_score(target, dataset['Cluster'], average='macro')
        logger.info(f'Jaccard Score: {jaccard}')

    if args.fm_score:
        fm = fowlkes_mallows_score(target, dataset['Cluster'])
        logger.info(f'Fowlkes-Mallows Score: {fm}')

    for cluster in range(optimal_clusters):
        cluster_data = dataset[dataset['Cluster'] == cluster]
        logger.info(f'Sumário do cluster {cluster}:')
        print(cluster_data.describe(include='all'))
        print('\n')


    if args.save_plot:
        plt.figure(figsize=(10, 7))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=dataset['Cluster'], cmap='viridis', marker='o', alpha=0.5)
        #plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], c='red', marker='x', s=200, label='Centroids')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('K-Means Clustering Visualization')
        plt.legend()
        plt.savefig(f'{args.output_path}/{args.clustering_algorithm}_clusters_{optimal_clusters}.png')
        plt.show()

if __name__ == '__main__':
    main()