from elasticsearch import Elasticsearch

def setup_es_client(host, api_key):
    return Elasticsearch(hosts=[host], api_key=api_key)

def delete_index(es_client, index_name):
    es_client.indices.delete(index=index_name, ignore_unavailable=True)