import pandas as pd

# Carregando o dataset
df = pd.read_csv('..\datasus\part-00000-0b7ee8fc-4d40-4b71-bec3-5d9ddbf54ec9.c000.csv', sep=';', encoding='utf-8')

# Exibindo as primeiras linhas do dataset
print(df.head())

# Exibindo todos os valores únicos da coluna 'sintomas'
sintomas = df['sintomas'].unique()
print(sintomas[0:10])

sintomas_unicos = []

# Coletando todos os sintomas únicos, por meio da separação dos sintomas por vírgula
for sintoma in sintomas:
    if isinstance(sintoma, str):
        sintomas_unicos.extend(sintoma.split(','))

# Removendo espaços em branco no início e no final de cada sintoma
sintomas_unicos = [sintoma.strip().lower() for sintoma in sintomas_unicos]
sintomas_unicos = list(set(sintomas_unicos))
sintomas_unicos.remove('')

print(sintomas_unicos)
print(len(sintomas_unicos))

# Listar todos os valores únicos por coluna
for coluna in df.columns:
    print(coluna, ":", df[coluna].unique())