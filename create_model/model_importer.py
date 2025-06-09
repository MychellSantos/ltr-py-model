from eland.ml import MLModel

def import_model(es_client, ranker, model_id, ltr_config):
    return MLModel.import_ltr_model(
        es_client=es_client,
        model=ranker,
        model_id=model_id,
        ltr_model_config=ltr_config,
        es_if_exists="replace"
    )
