from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.datatypes import convert
from clearml import Task
import warnings
import pandas as pd
import time

def tde(X_train, y_train, X_test, y_test):
    """
    Treina e avalia o classificador TemporalDictionaryEnsemble.

    Args:
        X_train (pd.DataFrame): Dados de treino no formato compatível com sktime.
        y_train (pd.Series): Rótulos de treino.
        X_test (pd.DataFrame): Dados de teste no formato compatível com sktime.
        y_test (pd.Series): Rótulos de teste.

    Returns:
        dict: Dicionário contendo as métricas de desempenho.
    """
    clf_TDE = TemporalDictionaryEnsemble(
        n_parameter_samples=250,
        max_ensemble_size=50,
        randomly_selected_params=50,
    ) 
    clf_TDE.fit(X_train, y_train) 
    tde_pred = clf_TDE.predict(X_test)
    
    # Calcular métricas com zero_division=0 para evitar avisos quando uma classe não for prevista
    metrics = {
        'accuracy_score': accuracy_score(y_test, tde_pred),
        'f1_score': f1_score(y_test, tde_pred, average='weighted', zero_division=0),
        'precision_score': precision_score(y_test, tde_pred, average='weighted', zero_division=0),
        'recall_score': recall_score(y_test, tde_pred, average='weighted', zero_division=0),
    }
    
    return metrics

def run_tde(
    clearML=True,
    params={
        'n_parameter_samples': 250,
        'max_ensemble_size': 50,
        'randomly_selected_params': 50,
        'k': 1,
        'K': 5,
        'country_train': 0,
        'country_test': 1,
        'city': 0,
        'category': None,
    },
    task=None,
    task_name="tde",
):
    """
    Executa o pipeline de treinamento e avaliação do classificador TemporalDictionaryEnsemble
    para ambos os cenários: treinar no país 0 e testar no país 1, e vice-versa.

    Args:
        clearML (bool): Se True, utiliza o ClearML para logging.
        params (dict): Dicionário de parâmetros para configuração.
        task (clearml.Task, optional): Objeto Task do ClearML.
        task_name (str): Nome da tarefa no ClearML.

    Returns:
        dict: Dicionário contendo os resultados das métricas para ambos os cenários.
    """
    start_time = time.time()

    if clearML:
        if task is None:
            task = Task.init(project_name='PopularTimesFold/Classifier', task_name=task_name)
        task.connect(params)

    # Carrega o dataset
    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')

    # Verifica se as colunas necessárias existem
    if 'country' not in df.columns or 'category' not in df.columns:
        raise ValueError("O CSV deve conter as colunas 'country' e 'category'.")

    # Separa os dados por país
    df_country_0 = df[df['country'] == 0].reset_index(drop=True)
    df_country_1 = df[df['country'] == 1].reset_index(drop=True)

    # Identifica as colunas que representam os pontos no tempo
    time_cols = [col for col in df.columns if col not in ['country', 'category']]

    # Verifica se há colunas de tempo
    if not time_cols:
        raise ValueError("Nenhuma coluna de tempo encontrada. Verifique as colunas do CSV.")

    # Função auxiliar para converter DataFrame para formato compatível com sktime
    def to_nested_univ(df, time_cols):
        """
        Converte um DataFrame tradicional em formato nested_univ esperado pela sktime.

        Args:
            df (pd.DataFrame): DataFrame contendo as séries temporais.
            time_cols (list): Lista de nomes das colunas que representam os pontos no tempo.

        Returns:
            pd.DataFrame: DataFrame no formato nested_univ.
        """
        return convert(df[time_cols], from_type="pd-flat", to_type="nested_univ")

    # Preparação dos dados para ambos os cenários
    scenarios = [
        {'train_country': 0, 'test_country': 1},
        {'train_country': 1, 'test_country': 0},
    ]

    all_results = {}

    for scenario in scenarios:
        train_country = scenario['train_country']
        test_country = scenario['test_country']

        # Seleciona os DataFrames de treino e teste
        if train_country == 0:
            df_train = df_country_0
            df_test = df_country_1
        else:
            df_train = df_country_1
            df_test = df_country_0

        # Converte para formato nested_univ
        X_train = to_nested_univ(df_train, time_cols)
        y_train = df_train['category']
        X_test = to_nested_univ(df_test, time_cols)
        y_test = df_test['category']

        # Nome do cenário
        scenario_name = f"train_country_{train_country}_test_country_{test_country}"
        print(f'Treinando no país {train_country} e testando no país {test_country}')

        # Executa a função de classificação
        main_time = time.time()
        results = tde(X_train, y_train, X_test, y_test)
        execution_time = time.time() - main_time

        # Armazena os resultados
        all_results[scenario_name] = results

        # Logging com ClearML, se ativado
        if clearML:
            task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=execution_time)
            for key, value in results.items():
                task.get_logger().report_scalar(
                    'metrics', 
                    f'{key}_{scenario_name}', 
                    iteration=0, 
                    value=value
                )

    # Tempo total de execução
    total_time = time.time() - start_time
    if clearML:
        task.get_logger().report_scalar('execution_time_total', 'main', iteration=0, value=total_time)
        task.close()

    return all_results

if __name__ == '__main__':
    results = run_tde()
    for scenario, metrics in results.items():
        print(f"\nResultados para {scenario}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")