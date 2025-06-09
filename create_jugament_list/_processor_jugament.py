from elasticsearch import helpers
from tqdm import tqdm
from _utils import is_core_categoria, calcula_grade
from config import SIZE_TO_MODEL

def fetch_search_results(es_client, source_index, search_value, size=SIZE_TO_MODEL):
    response = es_client.search(
        index=source_index,
        size=size,
        _source=['nome', '_id', 'precoSkuDe', 'popularidade', 'nomeCategoria',
                 'idCategoria', 'maisVendidos', 'classificacaoMedia'],
        query={
            "multi_match": {
                "fields": ["nome", "nomeCategoria"],
                "query": search_value
            }
        }
    )
    return response['hits']['hits']

def create_actions(hits, search_value, query_id, destination_index):
    actions = []
    for hit in hits:
        _source = hit['_source']
        document_id = hit['_id']
        document_score = hit['_score']
        category_id = _source.get("idCategoria", [])[-1] if _source.get("idCategoria") else 0
        action = {
            "_op_type": "index",
            "_index": destination_index,
            "_id": document_id,
            "_source": {
                "query_id": query_id,
                "query": search_value,
                "nome_text": _source.get('nome', "Desconhecido"),
                "doc_id": document_id,
                "grade": calcula_grade(is_core_categoria(category_id), _source),
                "price": _source.get('precoSkuDe', 0),
                "popularity": _source.get('popularidade', 0),
                "best_sellers": _source.get('maisVendidos', 0),
                "category": _source.get('nomeCategoria', "Desconhecido"),
                "category_id": category_id,
                "query_len": len(search_value.split(" ")),
                "classificacaoMedia": _source.get('classificacaoMedia', 0),
                "nome_score": str(document_score)
            }
        }
        actions.append(action)
    return actions

def bulk_index_with_progress(es_client, actions, query_id):
    with tqdm(total=len(actions), desc=f'Indexando query {query_id}', leave=False) as pbar:
        for success, info in helpers.streaming_bulk(es_client, actions):
            pbar.update()
            if not success:
                print('Erro ao indexar documento:', info)

def process_queries(es_client, df, source_index, destination_index):
    for query_id, row in enumerate(df.itertuples(index=False), 1):
        search_value = row.query.strip()
        hits = fetch_search_results(es_client, source_index, search_value)
        actions = create_actions(hits, search_value, query_id, destination_index)
        if actions:
            bulk_index_with_progress(es_client, actions, query_id)