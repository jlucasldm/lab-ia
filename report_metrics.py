import pickle
import argparse

def setup_argparser():
    parser = argparse.ArgumentParser(description='Load metrics from a pickle file')
    parser.add_argument('--file-path', type=str, required=True, help='Pickle file path')
    return parser

def carregar_metricas(pickle_file: str) -> dict:
    with open(pickle_file, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def relatar_metricas(metricas: dict):
    print(f'Acurácia: {metricas["accuracy"]}')
    print(f'Precisão: {metricas["precision"]}')
    print(f'Recall: {metricas["recall"]}')
    print(f'F1-Score: {metricas["f1"]}')
    print('Matriz de confusão:')
    print(metricas["conf_matrix"])
    print('Relatório de classificação:')
    print(metricas["class_report"])

def main():
    argparser = setup_argparser()
    args = argparser.parse_args()
    metrics = carregar_metricas(args.file_path)
    relatar_metricas(metrics)

if __name__ == '__main__':
    main()