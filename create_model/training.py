import matplotlib.pyplot as plt
from xgboost import  plot_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score
import numpy as np

def split_data(X, y, groups):
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7)
    train_idx, eval_idx = next(gss.split(X, y, groups))
    return X.loc[train_idx], X.loc[eval_idx], y.loc[train_idx], y.loc[eval_idx], groups.loc[train_idx], groups.loc[eval_idx]

def train_ranker(ranker, train_X, train_y, train_groups, eval_X, eval_y, eval_groups):
    ranker.fit(
        X=train_X,
        y=train_y,
        group=train_groups.value_counts().sort_index().values,
        eval_set=[(eval_X, eval_y)],
        eval_group=[eval_groups.value_counts().sort_index().values],
        verbose=True,
    )
    plot_importance(ranker, importance_type="weight")
    plt.show()

def validate_model_predictions(ranker, X_test, y_test, group_test):
    predictions = ranker.predict(X_test)
    ndcg_scores = []
    for group in np.unique(group_test):
        mask = group_test == group
        true_rel = y_test[mask].values
        pred = predictions[mask]
        ndcg = ndcg_score([true_rel], [pred], k=10)
        ndcg_scores.append(ndcg)
    print(f"NDCG médio: {np.mean(ndcg_scores):.4f}")
    return {"mean_ndcg": np.mean(ndcg_scores)}

def test_model_with_input(ranker, input_features, judgments_df):
    from collections import defaultdict
    predictions = ranker.predict(input_features)
    selected_columns = ['query_id', 'query', 'nome_text', 'doc_id', 'grade', 'nome_score']
    grouped = defaultdict(list)

    for idx in range(len(predictions)):
        row = judgments_df.loc[idx, selected_columns]
        row['nome_text'] = str(row['nome_text'])[:25]
        grouped[row['query_id']].append((row.to_list(), predictions[idx]))

    # Pega o primeiro grupo (query_id)
    first_query_id = next(iter(grouped))  # ou: list(grouped.keys())[0]

    print(f"\nTop 10 resultados para query_id={first_query_id}:")
    for data, peso in sorted(grouped[first_query_id], key=lambda x: x[1], reverse=True)[:10]:
        print(f"{data} | => peso = {peso:.4f}")
    return predictions

def save_predictions_to_elasticsearch(predictions, judgments_df, es_client, index_name, predict_field="peso_predict"):
    bulk_actions = []
    for idx, peso in enumerate(predictions):
        doc_id = judgments_df.loc[idx, 'doc_id']
        bulk_actions.append({
            "_op_type": "update",
            "_index": index_name,
            "_id": doc_id,
            "doc": {predict_field: float(peso)}
        })
    if bulk_actions:
        from elasticsearch.helpers import bulk
        success, _ = bulk(es_client, bulk_actions)
        print(f"✅ {success} documentos atualizados com {predict_field}")