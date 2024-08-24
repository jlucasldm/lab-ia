import argparse
import os
import pandas as pd
import pickle
import json
from venv import logger
from logging import getLogger, StreamHandler, INFO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def setup_argparser():
    parser = argparse.ArgumentParser(description='Treinar um modelo de decisão')
    parser.add_argument('--dataset-path', type=str, required=True, help='Caminho do dataset de input')
    parser.add_argument('--output-path', type=str, required=True, help='Diretório de output para os modelos treinados')
    parser.add_argument('--test-size', type=float, default=0.2, help='Tamanho do dataset de teste em porcentagem')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--target-label', type=str, default='classificacaoFinal', help='Nome da coluna alvo')
    parser.add_argument('--use-smote', action='store_true', help='Usar SMOTE para balancear as classes')
    parser.add_argument('--silent', action='store_true', help='Não exibir mensagens de log')
    parser.add_argument('--grid-search', action='store_true', help='Usar GridSearchCV para encontrar os melhores parâmetros')
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

def calcular_metricas(args: argparse.Namespace, test, model: DecisionTreeClassifier):
    y_pred = model.predict(test.drop(columns=[args.target_label]))
    y_true = test[args.target_label]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }
    return metrics

def relatar_metricas(metricas: dict):
    logger.info(f'Acurácia: {metricas["accuracy"]}')
    logger.info(f'Precisão: {metricas["precision"]}')
    logger.info(f'Recall: {metricas["recall"]}')
    logger.info(f'F1-Score: {metricas["f1"]}')
    logger.info('Matriz de confusão:')
    logger.info(metricas["conf_matrix"])
    logger.info('Relatório de classificação:')
    logger.info(metricas["class_report"])

def main():
    parser = setup_argparser()
    args = parser.parse_args()
    logger = setup_logger(args.silent)
    logger.info(f'Carregando dataset com caminho {args.dataset_path}')
    dataset = pd.read_csv(args.dataset_path)
    
    if args.use_smote:
        logger.info('Aplicando SMOTE para balancear as classes')
        smote = SMOTE(random_state=args.random_state, k_neighbors=2)
        X, y = smote.fit_resample(dataset.drop(columns=[args.target_label]), dataset[args.target_label])
        dataset = pd.concat([X, y], axis=1)
        print(dataset.info())
        print(dataset[args.target_label].value_counts())

    logger.info(f'Separando dataset em treino e teste com proporção de {args.test_size} para teste')
    train, test = train_test_split(dataset, test_size=args.test_size)

    grid_de_parametros = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_leaf_nodes': [None, 10, 20],
        'splitter': ['best', 'random']
    }

    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, grid_de_parametros, cv=5, n_jobs=-1, verbose=3)

    logger.info('Treinando modelo')
    melhor_modelo = None

    if args.grid_search:
        for _ in tqdm(range(1), desc="Progresso do GridSearchCV"):
            grid_search.fit(train.drop(columns=[args.target_label]), train[args.target_label])
        melhor_modelo = grid_search.best_estimator_
    else:
        for _ in tqdm(range(1), desc="Progresso do treinamento"):
            model.fit(train.drop(columns=[args.target_label]), train[args.target_label])
        melhor_modelo = model

    metricas = calcular_metricas(args, test, melhor_modelo)
    relatar_metricas(metricas)

    output_filename = f'dt_{args.test_size}_{'smote' if args.use_smote else 'no smote'}'
    output_path = os.path.join(args.output_path, output_filename)
    logger.info(f'Salvando modelo em {output_path}')

    with open(f'{output_path}.pkl', 'wb') as f:
        pickle.dump('melhor_modelo', f)
    
    with open(f'{output_path}_metricas.pkl', 'wb') as f:
        pickle.dump(metricas, f)

if __name__ == '__main__':
    main()