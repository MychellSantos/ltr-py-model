import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from es_client import setup_es_client, delete_index
from create_jugament_list._processor_jugament import process_queries
from config import ES_HOST, ES_KEY, SOURCE_INDEX, DESTINATION_INDEX, QUERY_FILE

def main():
    es_client = setup_es_client(ES_HOST, ES_KEY)
    delete_index(es_client, DESTINATION_INDEX)
    df = pd.read_csv(QUERY_FILE)
    process_queries(es_client, df, SOURCE_INDEX, DESTINATION_INDEX)
    print("Indexação concluída.")

if __name__ == "__main__":
    main()