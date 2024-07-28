import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from typing import Union

from typing import Dict, List, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
# Definindo os parâmetros globais
plt.rcParams.update({
    # 'font.family': 'serif',         # Fonte geral
    # 'font.serif': 'Times New Roman',# Estilo da fonte
    'font.size': 12,                # Tamanho da fonte
    'axes.labelsize': 12,           # Tamanho da fonte dos rótulos dos eixos
    'axes.titlesize': 12,           # Tamanho da fonte dos títulos dos subplots
    'xtick.labelsize': 10,          # Tamanho da fonte dos rótulos do eixo x
    'ytick.labelsize': 10,          # Tamanho da fonte dos rótulos do eixo y
    'legend.fontsize': 10,          # Tamanho da fonte da legenda
    'figure.titlesize': 10,         # Tamanho da fonte do título da figura
    # 'axes.spines.right': False,     # Remover a espinha do lado direito
    # 'axes.spines.top': False        # Remover a espinha do topo
    'savefig.dpi': 300,             # DPI para salvar a figura
    'savefig.format': 'png',        # Formato da figura ao salvar
    'savefig.bbox': 'tight',        # Remove espaços em branco extras
    'savefig.pad_inches': 0.1       # Espaçamento de preenchimento em polegadas
})

def calculate_auc_roc(df: DataFrame) -> float:
    """
    Calcula a AUC ROC para um DataFrame com colunas 'label' e 'prediction'.

    Args:
        df (DataFrame): DataFrame com as colunas 'label' e 'prediction'.

    Returns:
        float: Valor da AUC ROC.
    """
    # Cria um avaliador para Binary Classification
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label", metricName="areaUnderROC")
    
    # Calcula a AUC ROC
    auc_roc = evaluator.evaluate(df)
    
    return auc_roc

def calculate_auc_pr(df: DataFrame) -> float:
    """
    Calcula a AUC PR para um DataFrame com colunas 'label' e 'prediction'.

    Args:
        df (DataFrame): DataFrame com as colunas 'label' e 'prediction'.

    Returns:
        float: Valor da AUC PR.
    """
    # Cria um avaliador para Binary Classification
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label", metricName="areaUnderPR")
    
    # Calcula a AUC PR
    auc_pr = evaluator.evaluate(df)
    
    return auc_pr

def calculate_ks(df: DataFrame) -> float:
    """
    Calcula o KS (Kolmogorov-Smirnov) para um DataFrame com colunas 'label' e 'prediction'.
    
    Args:
        df (DataFrame): DataFrame com as colunas 'label' e 'prediction'.

    Returns:
        float: Valor do KS.
    """
    # Ordenar o DataFrame pelas previsões
    df_sorted = df.orderBy(F.col("prediction").desc())
    
    # Calcular o número total de positivos e negativos
    total_positives = df_sorted.filter(F.col("label") == 1).count()
    total_negatives = df_sorted.filter(F.col("label") == 0).count()
    
    # Adicionar colunas de contagem acumulada
    window_spec = Window.orderBy(F.col("prediction").desc())
    df_sorted = df_sorted.withColumn("cum_positives", F.sum(F.when(F.col("label") == 1, 1).otherwise(0)).over(window_spec))
    df_sorted = df_sorted.withColumn("cum_negatives", F.sum(F.when(F.col("label") == 0, 1).otherwise(0)).over(window_spec))
    
    # Calcular taxas acumuladas
    df_sorted = df_sorted.withColumn("tpr", F.col("cum_positives") / total_positives)
    df_sorted = df_sorted.withColumn("fpr", F.col("cum_negatives") / total_negatives)
    
    # Calcular KS
    df_sorted = df_sorted.withColumn("ks", F.col("tpr") - F.col("fpr"))
    ks_value = df_sorted.agg(F.max(col("ks"))).collect()[0][0]
    
    return ks_value

def calculate_confusion_matrix(df: DataFrame) -> Dict[str, int]:
    """
    Calcula os valores de verdadeiro positivo (TP), verdadeiro negativo (TN),
    falso positivo (FP) e falso negativo (FN) para um DataFrame com colunas 'label' e 'prediction'.

    Args:
        df (DataFrame): DataFrame com as colunas 'label' e 'prediction'.

    Returns:
        dict: Um dicionário com os valores de TP, TN, FP e FN.
    """
    tp = df.filter((F.col('label') == 1) & (F.col('prediction') == 1)).count()
    tn = df.filter((F.col('label') == 0) & (F.col('prediction') == 0)).count()
    fp = df.filter((F.col('label') == 0) & (F.col('prediction') == 1)).count()
    fn = df.filter((F.col('label') == 1) & (F.col('prediction') == 0)).count()
    
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def calcula_mostra_matriz_confusao(df_transform_modelo: DataFrame, normalize: bool = False, percentage: bool = True) -> None:
    """
    Calcula e exibe a matriz de confusão para um DataFrame com colunas 'label' e 'prediction'.

    Args:
        df_transform_modelo (DataFrame): DataFrame com as colunas 'label' e 'prediction'.
        normalize (bool): Se True, normaliza os valores pela soma das linhas. Default é False.
        percentage (bool): Se True, exibe os valores normalizados em percentual. Requer normalize=True. Default é True.

    Returns:
        None
    """
    tp = df_transform_modelo.select('label', 'prediction').where((F.col('label') == 1) & (F.col('prediction') == 1)).count()
    tn = df_transform_modelo.select('label', 'prediction').where((F.col('label') == 0) & (F.col('prediction') == 0)).count()
    fp = df_transform_modelo.select('label', 'prediction').where((F.col('label') == 0) & (F.col('prediction') == 1)).count()
    fn = df_transform_modelo.select('label', 'prediction').where((F.col('label') == 1) & (F.col('prediction') == 0)).count()
  
    valorP = 1
    valorN = 1

    if normalize:
        valorP = tp + fn
        valorN = fp + tn
  
    if percentage and normalize:
        valorP = valorP / 100
        valorN = valorN / 100

    print(' ' * 20, 'Previsto')
    print(' ' * 15, 'Churn', ' ' * 5, 'Não-Churn')
    print(' ' * 4, 'Churn', ' ' * 6, int(tp / valorP), ' ' * 7, int(fn / valorP))
    print('Real')
    print(' ' * 4, 'Não-Churn', ' ' * 2, int(fp / valorN), ' ' * 7, int(tn / valorN))


def bootstrap_metric_spark(
    data: DataFrame,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, List[float]]]:
    """
    Calcula o intervalo de confiança e a média para várias métricas usando o método de bootstrap.

    Args:
    - data: DataFrame do Spark contendo os dados com as colunas 'label' e 'prediction'.
    - n_bootstrap: Número de amostras bootstrap a serem geradas (padrão é 1000).
    - confidence_level: Nível de confiança para o intervalo de confiança (padrão é 0.95).

    Returns:
    - Um dicionário onde as chaves são os nomes das métricas ('ks', 'auc', 'auc_pr') e os valores são dicionários contendo:
      - 'scores': Lista de pontuações para a métrica.
      - 'interval': Limites inferior e superior do intervalo de confiança.
      - 'mean_score': Média das pontuações calculadas nas amostras bootstrap.
      - 'std_dev': Desvio padrão das pontuações calculadas nas amostras bootstrap.
    """
    bootstrapped_scores_ks = []
    bootstrapped_scores_auc_roc = []
    bootstrapped_scores_auc_pr = []
    # Inicializa o gerador de números aleatórios para garantir reprodutibilidade
    rng = np.random.RandomState(42)
    print(f'Será realizada {n_bootstrap} iterações')
    for i in range(n_bootstrap):
        print(f'Execução iteração: {i}')
        # Reamostragem com substituição
        sample = data.sample(withReplacement=True, fraction=1.0, seed=rng.randint(1, 10000))
        # `withReplacement` é True, cada linha do DataFrame pode ser escolhida mais de uma vez na amostra.
        # `fraction=1.0` Um valor de 1.0 significa que a amostra deve ter o mesmo número de linhas que o DataFrame original,
        # se fosse 0.5, a amostra teria aproximadamente 50% das linhas do DataFrame original.
        # seed=rng.randint(1, 10000): Usando a abordagem com rng, você pode obter uma nova semente aleatória para cada iteração.

        print(f'Sample count: {sample.count()}')
        
        # Cálculo da métrica
        score_ks = calculate_ks(sample)
        score_auc_roc = calculate_auc_roc(sample)
        score_auc_pr = calculate_auc_pr(sample)
        
        bootstrapped_scores_ks.append(score_ks)
        bootstrapped_scores_auc_roc.append(score_auc_roc)
        bootstrapped_scores_auc_pr.append(score_auc_pr)
        print('---'*5)
    
    # Lista contendo as listas de pontuações e suas respectivas chaves
    listas = [
        ('ks', bootstrapped_scores_ks),
        ('auc', bootstrapped_scores_auc_roc),
        ('auc_pr', bootstrapped_scores_auc_pr)
    ]
    resultados = {}
    resultados_scores = {}
    # Iterar sobre cada lista e calcular os valores desejados
    for chave, scores in listas:
        sorted_scores = np.array(scores)
        lower_bound = float(np.percentile(sorted_scores, (1 - confidence_level) / 2 * 100))
        upper_bound = float(np.percentile(sorted_scores, (1 + confidence_level) / 2 * 100))
        mean_score = float(np.mean(sorted_scores))
        std_dev = float(np.std(sorted_scores, ddof=1))  # Usando ddof=1 para amostras

        confidence_interval = [lower_bound, upper_bound]
    
        resultados[chave] = {
            'confidence_interval': confidence_interval,
            'mean_score': mean_score,
            'std_dev': std_dev
        }

        resultados_scores[chave] = {
            'scores': scores,
        }
    return resultados_scores, resultados


def df_scores(scores_dic: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """
    Converte um dicionário de scores em um DataFrame do Pandas.

    Args:
        scores_dic (dict): Dicionário contendo os scores. O formato esperado é:
            {
                'ks': {'scores': list},
                'auc': {'scores': list},
                'auc_pr': {'scores': list}
            }

    Returns:
        pd.DataFrame: DataFrame contendo as listas de scores com as seguintes colunas:
            - 'ks.scores': Scores de KS.
            - 'auc.scores': Scores de AUC.
            - 'auc_pr.scores': Scores de AUC-PR.
    """
    df = pd.DataFrame()
    df['ks.scores'] = scores_dic['ks']['scores']
    df['auc.scores'] = scores_dic['auc']['scores']
    df['auc_pr.scores'] = scores_dic['auc_pr']['scores']
    return df

### PERMUTACION TESTE

def permutation_test(
    array1: List[float],
    array2: List[float],
    anscreen: bool = False,
    alpha: float = 0.05
) -> Tuple[float, List[float], float, List[str]]:
    """
    Realiza um teste de permutação para comparar as médias de dois arrays.

    Args:
        array1 (List[float]): O primeiro array de dados.
        array2 (List[float]): O segundo array de dados.
        anscreen (bool): Se True, imprime os resultados na tela. Default é False.
        alpha (float): Nível de significância para o teste (p-valor). Default é 0.05.

    Returns:
        Tuple[float, List[float], float, List[str]]:
            - p_val (float): Valor p do teste de permutação.
            - mean_lst (List[float]): Lista das diferenças médias permutadas.
            - mean_diff (float): Diferença média observada entre os dois arrays.
            - text_lst (List[str]): Lista de mensagens interpretativas sobre o teste.
    """
    # Garantindo a entrada com numpy array
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    # Cálculo das médias de cada vetor
    avg_array1 = array1.mean()
    avg_array2 = array2.mean()
    
    # Diferença entre as médias
    mean_diff = avg_array1 - avg_array2
    full_array = np.concatenate([array1, array2])
    mean_lst = []
    # Defina a semente aleatória para reprodutibilidade
    np.random.seed(42)
    for i in range(10000):
        # Com reposição: bootstrapping
        avg1 = np.random.choice(full_array, size=len(array1), replace=True).mean()
        avg2 = np.random.choice(full_array, size=len(array2), replace=True).mean()
        # reprece = True, Assume que qualquer valor pode vir de uma das duas listas, convergÊncia para Normal.
        mean_lst.append(avg1 - avg2)
    
    if mean_diff > 0:
        p_val = np.sum(np.array(mean_lst) > mean_diff) / len(mean_lst) #ou 10000
    else:
        p_val = np.sum(np.array(mean_lst) < mean_diff) / len(mean_lst) #ou 10000
    
    text_lst = ["\n Teste de Significancia ", 
                "**$H_0$:** Diferença entre as médias das métricas é zero. \n",
                f" Arrays sizes: {len(array1)}, {len(array2)} ",
                "* Difference between averages: %.4f - %.4f = %.4f" % (avg_array1, avg_array2, mean_diff),
                "* p_val = %.4f " %p_val]
    
    if p_val > alpha:
        text_lst.append(f'The model seems to produce similar results with CI-{1 - alpha} (fail to reject H0).\n')
    else:
        text_lst.append(f'The model seems to produce different results with CI-{1 - alpha} (reject H0).\n')
    
    if anscreen:
        for line in text_lst:
            print(line)   
    return p_val, mean_lst, mean_diff, text_lst

def bootstrap_metric_spark_permutacion(
    data1: DataFrame,
    data2: DataFrame,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, float]]]:
    """
    Calcula o intervalo de confiança e a média para várias métricas usando o método de bootstrap e realiza um teste de permutação para comparar as métricas entre dois DataFrames.

    Args:
        data1 (DataFrame): Primeiro DataFrame do Spark contendo os dados com as colunas 'label' e 'prediction'.
        data2 (DataFrame): Segundo DataFrame do Spark contendo os dados com as colunas 'label' e 'prediction'.
        n_bootstrap (int): Número de amostras bootstrap a serem geradas. Default é 100.
        confidence_level (float): Nível de confiança para o intervalo de confiança. Default é 0.95.

    Returns:
        Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, float]]]:
            - resultados_scores (Dict[str, Dict[str, List[float]]]): Dicionário com os scores bootstrap para cada métrica.
            - resultados (Dict[str, Dict[str, float]]): Dicionário com os intervalos de confiança, médias e desvios padrão das métricas.
            - p_values (Dict[str, float]): Dicionário com os valores p dos testes de permutação para cada métrica.
    """
    def calculate_metrics(data: DataFrame) -> Dict[str, float]:
        """Calcula as métricas de desempenho para um DataFrame."""
        return {
            'ks': calculate_ks(data),
            'auc': calculate_auc_roc(data),
            'auc_pr': calculate_auc_pr(data)
        }

    bootstrapped_scores1 = {metric: [] for metric in ['ks', 'auc', 'auc_pr']}
    bootstrapped_scores2 = {metric: [] for metric in ['ks', 'auc', 'auc_pr']}
    
    rng = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    # está gerando várias amostras bootstrap a partir de data1 e data2
    for i in range(n_bootstrap):
        print(f'Iteração {i+1}/{n_bootstrap}')
        #withReplacement=True: Indica que a amostragem é com reposição (Bootstrapping)
        sample1 = data1.sample(withReplacement=True, fraction=1.0, seed=rng.randint(1, 10000))
        sample2 = data2.sample(withReplacement=True, fraction=1.0, seed=rng2.randint(1, 10000))
        print(f'Sample1 count: {sample1.count()}')
        print(f'Sample2 count: {sample2.count()}')
        print('--'*5)
        
        metrics1 = calculate_metrics(sample1)
        metrics2 = calculate_metrics(sample2)
        
        for metric in ['ks', 'auc', 'auc_pr']:
            bootstrapped_scores1[metric].append(metrics1[metric])
            bootstrapped_scores2[metric].append(metrics2[metric])
    
  
    results = {}
    results_scores_permutacion = {}
    
    for metric in ['ks', 'auc', 'auc_pr']:
        scores1 = np.array(bootstrapped_scores1[metric])
        scores2 = np.array(bootstrapped_scores2[metric])
        #### Cálculo do intervalo de confiança
        std_dev1 = np.std(scores1, ddof =1)
        mean_score1 = np.mean(scores1)
        lower_bound1, upper_bound1 = st.t.interval(
            # alpha=alpha, # Versão antiga so scipy
            confidence=confidence_level,
            df=len(scores1) -1,
            loc=mean_score1,
            scale=st.sem(scores1))
        
        #### Cálculo do intervalo de confiança
        std_dev2 = np.std(scores2, ddof =1)
        mean_score2 = np.mean(scores2)
        lower_bound2, upper_bound2 = st.t.interval(
            # alpha=alpha, # Versão antiga so scipy
            confidence=confidence_level,
            df=len(scores2) -1,
            loc=mean_score2,
            scale=st.sem(scores2))
        
        results[metric] = {
            'confidence_interval1': [lower_bound1, upper_bound1],
            'mean_score1': mean_score1,
            'std_dev1': std_dev1,
            'confidence_interval2': [lower_bound2, upper_bound2],
            'mean_score2': mean_score2,
            'std_dev2': std_dev2
        }
        
        # Teste de permutação entre os dois conjuntos de scores
        # Cada elemento dos scores, foi gerado por amostras com reposição (Bootstrapping )
        p_val, mean_lst, mean_diff, text_lst = permutation_test(scores1.tolist(), scores2.tolist())
        print('####'*10)
        print(metric)
        plot_pts(
            data=mean_lst, 
            p_valor=p_val, 
            mean_diff=mean_diff,
            metric = metric
            )
        plt.show()
        for line in text_lst:
            print(line) 
        
        results_scores_permutacion[metric] = {
            'scores1': scores1.tolist(),
            'scores2': scores2.tolist(),
            'p_value': p_val,
            'mean_diff': mean_diff
        }
    
    return results_scores_permutacion, results

def df_scores_1_2(scores_dic: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """
    Converte um dicionário de scores em um DataFrame do Pandas.

    Args:
        scores_dic (dict): Dicionário contendo os scores1 e scores2.

    Returns:
        pd.DataFrame: DataFrame contendo as listas de scores e informações de teste com as seguintes colunas:
            - 'ks.scores1': Scores de KS para o primeiro conjunto.
            - 'ks.scores2': Scores de KS para o segundo conjunto.
            - 'auc.scores1': Scores de AUC para o primeiro conjunto.
            - 'auc.scores2': Scores de AUC para o segundo conjunto.
            - 'auc_pr.scores1': Scores de AUC-PR para o primeiro conjunto.
            - 'auc_pr.scores2': Scores de AUC-PR para o segundo conjunto.
    """
    df = pd.DataFrame()
    df['ks.scores1'] = scores_dic['ks']['scores1']
    df['ks.scores2'] = scores_dic['ks']['scores2']
    print(f"KS p_value': {scores_dic['ks']['p_value']}")
    print(f"KS mean_diff': {scores_dic['ks']['mean_diff']}")
    
    df['auc.scores1'] = scores_dic['auc']['scores1']
    df['auc.scores2'] = scores_dic['auc']['scores2']
    print(f"auc p_value: {scores_dic['auc']['p_value']}")
    print(f"auc mean_diff: {scores_dic['auc']['mean_diff']}")
    
    df['auc_pr.scores1'] = scores_dic['auc_pr']['scores1']
    df['auc_pr.scores2'] = scores_dic['auc_pr']['scores2']
    print(f"auc_pr p_value: {scores_dic['auc_pr']['p_value']}")
    print(f"auc_pr mean_diff: {scores_dic['auc_pr']['mean_diff']}")
    return df


def plot_pts(
        data: np.ndarray, 
        p_valor: Union[float, int], 
        mean_diff: float,
        metric:str,
    ) -> plt.Figure:
    """
    Plota um histograma para visualizar o resultado de um teste de permutação.

    Args:
        data (np.ndarray): Os dados a serem plotados, representando as diferenças de média geradas durante o teste de permutação.
        p_valor (float): O valor p calculado do teste de permutação.
        mean_diff (float): A diferença de médias observada entre os grupos.
        metric (str): Nome da metrica a ser plotada.

    Returns:
        plt.Figure: Um objeto matplotlib.pyplot que contém o histograma do teste de permutação.
    """
    
    # Cria o histograma com um número fixo de bins
    # O parâmetro `bins` define o número de subdivisões do histograma
    ni, xi = np.histogram(data, bins=17)

    # Normaliza o histograma para que a soma das frequências seja 1
    fi = ni / np.sum(ni)

    # Calcula a largura do intervalo dos bins
    delta_xi = xi[1] - xi[0]
    xi = xi + np.abs(delta_xi) / 2.0
    print("* Intervalo do bin: %.3f" % delta_xi)

    # Plota a linha vertical indicando a diferença de médias observada
    plt.plot(
        [mean_diff, mean_diff], 
        [0, np.max(ni) * 0.85], 
        color='red', 
        linestyle='--', 
        label='Diff. observada'
    )

    # Define o alinhamento do texto dependendo do valor de p_valor e mean_diff
    if p_valor <= 0.05:
        alinhamento = 'center'
    else:
        if mean_diff < 0:
            alinhamento = 'right'
        else:
            alinhamento = 'left'

    # Adiciona texto ao plot com o valor p
    plt.text(mean_diff, np.max(ni) * 0.9, f'p_val={np.round(p_valor, 3)}',
             horizontalalignment=alinhamento, fontsize=12)

    # Plota o histograma
    plt.bar(xi[:-1], ni, label='Data', edgecolor='black', color=[0, 0.6, 0.3], 
            width=np.abs(delta_xi) * 0.9, align='center')

    # Configurações do plot
    plt.title(f"Permutation Test Score - {metric}")
    plt.xlabel("Difference of means")
    plt.ylabel("Counts")
    plt.legend()
    plt.grid()
    plt.savefig(f"work/output/figures/{metric}_permutation_test.png", dpi=300, format="png", bbox_inches="tight", pad_inches=0.1)

    # Retorna o objeto plt para permitir manipulação adicional
    return plt

