from elasticsearch import Elasticsearch, helpers
import numpy as np
import pandas as pd
from tqdm import tqdm

def setup_es_client(host, api_key):
    return Elasticsearch(hosts=[host], api_key=api_key)

def delete_index(es_client, index_name):
    es_client.indices.delete(index=index_name, ignore_unavailable=True)

def fetch_search_results(es_client, source_index, search_value, size=50):
    response = es_client.search(
        index=source_index,
        size=size,
        _source=['_id','precoSkuDe','popularidade' ,'nomeCategoria'],
        query={
            "match": {
                "nome": {
                    "query":search_value
                }
            }
        }
    )
    return response['hits']['hits']

def create_actions(hits, search_value, query_id, destination_index):
    numeros = [1, 2, 3, 4]
    probabilidades = [0.33, 0.33, 0.33, 0.01] 
    actions = []
    for hit in hits:
        document_id = hit['_id']
        document_score = hit['_score']
        _source = hit['_source']
        action = {
            "_op_type": "index",
            "_index": destination_index,
            "_id": document_id,
            "_source": {
                "query_id": query_id,
                "query": search_value,
                "doc_id": document_id,
                "grade": np.random.choice(numeros, p=probabilidades),
                "price": _source.get('precoSkuDe', 0),
                "popularity": _source.get('popularidade', 0), 
                "category": _source.get('nomeCategoria', "Desconhecido"),
                "query_len": len(search_value.split(" ")),
                "nome_score": document_score
            }
        }
        actions.append(action)
    return actions

def bulk_index_with_progress(es_client, actions, query_id):
    with tqdm(total=len(actions), desc=f'Indexando query {query_id}', leave=False) as pbar:
        def bulk_progress_wrapper():
            for success, info in helpers.streaming_bulk(es_client, actions):
                pbar.update()
                if not success:
                    print('Erro ao indexar documento:', info)
        bulk_progress_wrapper()

def process_queries(es_client, df, source_index, destination_index):
    query_id = 1
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processando pesquisas'):
        search_value = row['query'].strip()
        hits = fetch_search_results(es_client, source_index, search_value)
        actions = create_actions(hits, search_value, query_id, destination_index)
        if actions:
            bulk_index_with_progress(es_client, actions, query_id)
        query_id += 1

def main():
    #region
    ES_HOST = ""
    ES_KEY = ""
    #endregion
    
    es_client = setup_es_client(ES_HOST, ES_KEY)
    
    source_index = 'search-product-ex-temp'
    destination_index = 'jugamento_lista'
    
    delete_index(es_client, destination_index)
    
    df = pd.read_csv("seed_query.csv")
    
    process_queries(es_client, df, source_index, destination_index)
    
    print("Indexação concluída.")

if __name__ == "__main__":
    main()
