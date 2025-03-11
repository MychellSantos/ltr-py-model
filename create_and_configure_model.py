import numpy
from tqdm import tqdm
import pandas as pd
from eland.ml.ltr import LTRModelConfig, QueryFeatureExtractor
from xgboost import XGBRanker, plot_importance
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
from eland.ml import MLModel
from getpass import getpass
from elasticsearch import Elasticsearch
from eland.ml.ltr import FeatureLogger
import time
from elasticsearch import Elasticsearch
import pandas as pd
from xgboost import XGBRanker, plot_importance
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import ndcg_score,average_precision_score


def setup_es_client(host, api_key):
    return Elasticsearch(hosts=[host], api_key=api_key)

def fetch_judgments(es_client, index_name, size=1000):
    print("Obtendo lista de julgamentos.")
    response = es_client.search(index=index_name, size=size, query={"match_all": {}})
    hits = response['hits']['hits']
    judgments_df = pd.DataFrame([hit['_source'] for hit in hits])
    return judgments_df


def _extract_query_features(query_judgements_group,feature_logger,ltr_config):
    doc_ids = query_judgements_group["doc_id"].astype("str").to_list()
    query_params = {"query": query_judgements_group["query"].iloc[0]}
    doc_features = feature_logger.extract_features(query_params, doc_ids)
    for feature_index, feature_name in enumerate(ltr_config.feature_names):
        query_judgements_group[feature_name] = numpy.array(
            [doc_features[doc_id][feature_index] for doc_id in doc_ids]
        )
    return query_judgements_group


def fetch_judgments_with_lambda(es_client, index_name, ltr_config):
    print("by fetch_judgments_with_lambda.")

    judgments_df = fetch_judgments(es_client,"jugamento_lista")
    print(judgments_df)
    feature_logger = FeatureLogger(es_client, index_name, ltr_config)
    judgments_with_features = judgments_df.groupby(
        "query_id", group_keys=False
    ).apply(lambda x: _extract_query_features(x, feature_logger,ltr_config))

    judgments_with_features

    return judgments_with_features


def split_data(X, y, groups):
    group_preserving_splitter = GroupShuffleSplit(n_splits=1, train_size=0.7).split(X, y, groups)
    train_idx, eval_idx = next(group_preserving_splitter)
    
    train_features, eval_features = X.loc[train_idx], X.loc[eval_idx]
    train_target, eval_target = y.loc[train_idx], y.loc[eval_idx]
    train_query_groups, eval_query_groups = groups.loc[train_idx], groups.loc[eval_idx]
    
    return train_features, eval_features, train_target, eval_target, train_query_groups, eval_query_groups

def train_ranker(ranker, train_features, train_target, train_query_groups, eval_features, eval_target, eval_query_groups):
    ranker.fit(
        X=train_features,
        y=train_target,
        group=train_query_groups.value_counts().sort_index().values,
        eval_set=[(eval_features, eval_target)],
        eval_group=[eval_query_groups.value_counts().sort_index().values],
        verbose=True,
    )
    plot_importance(ranker, importance_type="weight")
    plt.show()

def import_model(es_client, ranker, model_id, ltr_config):
     MLModel.import_ltr_model(
        es_client=es_client,
        model=ranker,
        model_id=model_id,
        ltr_model_config=ltr_config,
        es_if_exists="replace"
    )
import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score

def validate_model_predictions(ranker, test_features, test_target, test_query_groups):
    """
    Valida o modelo de ranking com métricas NDCG e MAP usando os dados de teste.

    Args:
        ranker: O modelo treinado (XGBRanker).
        test_features: Features do conjunto de teste (DataFrame).
        test_target: Valores reais de relevância (Series).
        test_query_groups: Grupos de consulta para teste (Series).

    Returns:
        dict: Métricas de avaliação como NDCG médio e MAP médio.
    """
    print("\nValidando o modelo...")

    # Fazer previsões
    test_predictions = ranker.predict(test_features)

    # Avaliação por grupos (query groups)
    unique_groups = test_query_groups.unique()
    ndcg_scores = []
    map_scores = []

    for group in unique_groups:
        group_mask = test_query_groups == group
        true_relevance = test_target[group_mask].values  # Relevância real
        predicted_scores = test_predictions[group_mask]  # Pontuações preditas

        # Criando arrays 2D para passar para as funções
        true_relevance_2d = true_relevance.reshape(1, -1)  # Transforma em 2D
        predicted_scores_2d = predicted_scores.reshape(1, -1)  # Transforma em 2D

        # Calcular NDCG para o grupo
        ndcg = ndcg_score(true_relevance_2d, predicted_scores_2d, k=10)
        ndcg_scores.append(ndcg)

        # Calcular MAP para o grupo
        #map_score = average_precision_score(true_relevance, predicted_scores)
        #map_scores.append(map_score)

    # Métricas médias
    mean_ndcg = np.mean(ndcg_scores)
    #mean_map = np.mean(map_scores)

    print(f"\nResultados de validação:")
    print(f"NDCG médio: {mean_ndcg:.4f}")
    #print(f"MAP médio: {mean_map:.4f}")

    return {"mean_ndcg": mean_ndcg}


def test_model_with_input(ranker, input_features, judgments_df):
    """
    Testa o modelo de ranking para um termo específico, mostrando a relevância prevista para o termo.

    Args:
        ranker: O modelo treinado (XGBRanker).
        input_features: Features do conjunto de teste (DataFrame).
        input_query_groups: Grupos de consulta (Series).
        input_query: O termo de consulta a ser testado.

    Returns:
        dict: Previsões do modelo para o termo de consulta.
    """
    # Previsão das relevâncias
    predictions = ranker.predict(input_features)

    # sorted_idx = np.argsort(predictions)[::-1]
    # predictions = predictions[sorted_idx]

    # Exibindo os resultados para o termo de consulta
    # print(f"\nPrevisões para o termo '{input_query}':")

    print("Resultado predict:")
    selected_columns = ['query_id', 'query', 'doc_id', 'grade','nome_score']
    for idx, item in judgments_df.iterrows():
       print(f"{item[selected_columns].to_list()} | peso = {predictions[idx]:.4f}")

    return predictions


def main():

    ES_HOST = ""
    ES_KEY = ""
    
    es_client = setup_es_client(ES_HOST, ES_KEY)
    
    ltr_config = LTRModelConfig(
        feature_extractors=[
            QueryFeatureExtractor(
                feature_name="nome_score", 
                query={"match": {"nome":"{{query}}"}}
            ),
            QueryFeatureExtractor(
            feature_name="nome_score_all_terms",
            query={
                "match": {
                    "nome": {"query": "{{query}}", "minimum_should_match": "100%"}
                }
            },
            ),
            QueryFeatureExtractor(
            feature_name="category_score_all_terms",
            query={
                "match": {
                    "category": {"query": "{{query}}", "minimum_should_match": "100%"}
                }
            },
            ),
            QueryFeatureExtractor(
            feature_name="popularity",
            query={
                "script_score": {
                    "query": {"exists": {"field": "popularity"}},
                    "script": {"source": "return doc['popularity'].value;"},
                }
            },
            ),
            QueryFeatureExtractor(
            feature_name="price",
            query={
                "script_score": {
                    "query": {"exists": {"field": "price"}},
                    "script": {"source": "return doc['price'].value;"},
                }
            },
            ),    
            QueryFeatureExtractor(
                feature_name="query_len",
                query={
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "return params['query'].splitOnToken(' ').length;",
                            "params": {
                                "query": "{{query}}",
                            }
                        },
                    }
                },
            )   
        ]
    )

    judgments_df = fetch_judgments_with_lambda(es_client,"search-product-ex-temp",ltr_config)

    #judgments_df = fetch_judgments(es_client,"jugamento_lista")

    print("Estrutura do DataFrame:")
    print(judgments_df)
    
    X = judgments_df[ltr_config.feature_names]
    y = judgments_df["grade"]
    groups = judgments_df["query_id"]
    
    train_features, eval_features, train_target, eval_target, train_query_groups, eval_query_groups = split_data(X, y, groups)
    
    ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric=["ndcg@10"],
        early_stopping_rounds=20,
    )
    
    train_ranker(ranker, train_features, train_target, train_query_groups, eval_features, eval_target, eval_query_groups)

    doc_counts = judgments_df.groupby('query_id').size()

    validation_metrics = validate_model_predictions(ranker, eval_features, eval_target, eval_query_groups)

    test_model_with_input(ranker, X.head(5), judgments_df.head(5))
    test_model_with_input(ranker, X.head(20), judgments_df.head(20))

    importar = input("Importar modelo? (s/n): ").strip().lower()
    
    
    if importar == "s":
        LEARNING_TO_RANK_MODEL_ID = "ltr-model-xgboost-custom"
        result = import_model(es_client, ranker, LEARNING_TO_RANK_MODEL_ID, ltr_config)
        print(result)

if __name__ == "__main__":
    main()
