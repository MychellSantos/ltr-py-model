import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from es_client import setup_es_client
from config import ES_HOST, ES_KEY, SOURCE_INDEX, DESTINATION_INDEX
from model_config import ltr_config
from data_loader import fetch_judgments_with_lambda
from training import split_data, train_ranker, test_model_with_input, save_predictions_to_elasticsearch
from model_importer import import_model
from xgboost import XGBRanker


def main():
    es_client = setup_es_client(ES_HOST, ES_KEY)
    judgments_df = fetch_judgments_with_lambda(es_client, SOURCE_INDEX, ltr_config)

    X = judgments_df[ltr_config.feature_names]
    y = judgments_df["grade"]
    groups = judgments_df["query_id"]

    train_X, eval_X, train_y, eval_y, train_groups, eval_groups = split_data(X, y, groups)

    ranker = XGBRanker(
        objective="rank:ndcg",
        eval_metric=["ndcg@20"],
        early_stopping_rounds=50,
    )

    train_ranker(ranker, train_X, train_y, train_groups, eval_X, eval_y, eval_groups)

    predictions = test_model_with_input(ranker, X, judgments_df)
    save_predictions_to_elasticsearch(predictions, judgments_df, es_client, DESTINATION_INDEX, "peso_predict")

    if input("Importar modelo? (s/n): ").strip().lower() == "s":
        try:
            model = import_model(es_client, ranker, "ltr-model-custom", ltr_config)
            print("✅ Modelo importado com sucesso!")
        except Exception as e:
            print("❌ Falha ao importar o modelo:")
            print(e)

if __name__ == "__main__": 
    main()