import pandas as pd
import numpy as np
from eland.ml.ltr import FeatureLogger
from config import SIZE_FETCH_DESTINATION, DESTINATION_INDEX

def fetch_judgments(es_client, index_name, size=SIZE_FETCH_DESTINATION):
    print("Obtendo lista de julgamentos.")
    response = es_client.search(index=index_name, size=size, query={"match_all": {}})
    hits = response['hits']['hits']
    return pd.DataFrame([hit['_source'] for hit in hits])

def _extract_query_features(query_judgements_group, feature_logger, ltr_config):
    doc_ids = query_judgements_group["doc_id"].astype("str").to_list()
    query_params = {"query": query_judgements_group["query"].iloc[0]}
    doc_features = feature_logger.extract_features(query_params, doc_ids)

    for feature_index, feature_name in enumerate(ltr_config.feature_names):
        query_judgements_group[feature_name] = np.array(
            [doc_features[doc_id][feature_index] for doc_id in doc_ids]
        )
    return query_judgements_group

def fetch_judgments_with_lambda(es_client, index_name, ltr_config):
    judgments_df = fetch_judgments(es_client, DESTINATION_INDEX)
    feature_logger = FeatureLogger(es_client, index_name, ltr_config)
    return judgments_df.groupby("query_id", group_keys=False).apply(
        lambda x: _extract_query_features(x, feature_logger, ltr_config)
    )
