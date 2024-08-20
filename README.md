# Classificação de Notificações de Síndrome Gripal

Trabalho final do curso de laboratório de Inteligência Artificial na Universidade Federal da Bahia. Foram desenvolvias duas soluções para o problema de classificação de [notificações de síndrome gripal](https://dados.gov.br/dados/conjuntos-dados/notificacoes-de-sindrome-gripal-leve-2024) do estado da bahia em 2024, uma utilizando um modelo não supervisionado e outra utilizando um modelo supervisionado.

## Descrição do dataset
### Definição: 
O Ministério da Saúde, por meio da Secretaria de Vigilância em Saúde e Ambiente (SVSA), implementou, devido à pandemia, a vigilância da Síndrome Gripal (SG) de casos leves e moderados suspeitos de covid-19. Esta página tem como finalidade disponibilizar a base de dados de SG de casos leves e moderados suspeitos de covid-19, a partir da incorporação do sistema e-SUS Notifica, em vigor a partir de março de 2020.

### Descrição da informação disponibilizada: 
Os dados são oriundos do sistema e-SUS Notifica, que foi desenvolvido para registro de casos de Síndrome Gripal suspeitos de covid-19, e contém dados referentes ao local de residência do paciente (variáveis: estado, município), independentemente de terem sido notificados em outro estado ou município (variáveis: estadoNotificação, municípioNotificação), resultados de exames laboratoriais, além de dados demográficos e clínicos epidemiológicos.

### Limitações dos dados: 
Estados e municípios que utilizam sistemas próprios de notificação de casos suspeitos de covid-19 estão em processo de integração com o e-SUS Notifica. Assim, os dados desses locais podem apresentar uma diferença substancial até a finalização do processo de integração. Devido ao grande número de casos notificados de SG suspeitos de covid-19, alguns estados já possuem mais de 1.048.576 de registros, o que impossibilita estes dados de serem abertos e analisados em planilha de Excel. Para maiores informações, consultar a Ficha de Investigação e o Instrutivo de preenchimento da ficha, disponíveis por meio no endereço eletrônico: https://datasus.saude.gov.br/notifica/

## Objetivo do Trabalho de Conclusão do Curso
Avaliar o desempenho de algoritmos de Aprendizado de Máquina para resolução de problemas práticos.

## Requisitos
* Selecionar uma base de dados com rótulo;
* Realizar experimentos usando algoritmos supervisionados e não-supervisionados vistos em sala de aula;
* Para executar a abordagem não-supervisionadada, suprimir os rótulos e validar os resultados usando
silhueta e índices externos;
* Realizar avaliação criteriosa dos resultados, justificando o uso de parâmetros e comparando os resultados
obtidos técnicas de validação;
* Justificar a escolha dos algoritmos e da validação utilizada.
* **Não serão aceitos trabalhos realizados sobre bases do UCI ou bases de dados da internet com aplicações
prévias de algoritmos de AM**

## Avaliação
* Execução do pré-processamento (2,5);
* Aplicação dos modelos de AM (2,5);
* Validação dos resultados (2,5);
* Criatividade na escolha do problema (2,5);