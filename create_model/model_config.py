from eland.ml.ltr import QueryFeatureExtractor, LTRModelConfig

ltr_config = LTRModelConfig(
    feature_extractors=[
            QueryFeatureExtractor(
                feature_name="nome_score__bm25_dismax",
                query={
                    "dis_max": {
                        "tie_breaker": 0.7,
                        "queries": [
                            {
                                "match_phrase": {
                                    "nome": {
                                        "boost": 10,
                                        "query": "{{query}}",
                                        "slop": 3
                                    }
                                }
                            },
                            {
                                "match": {
                                    "nome": {
                                        "fuzziness": "AUTO",
                                        "operator": "and",
                                        "query": "{{query}}"
                                    }
                                }
                            },
                            {
                                "match_phrase": {
                                    "nomesAlternativos": {
                                        "boost": 9,
                                        "query": "{{query}}",
                                        "slop": 3
                                    }
                                }
                            },
                            {
                                "match": {
                                    "nomesAlternativos": {
                                        "boost": 7,
                                        "fuzziness": "AUTO",
                                        "operator": "and",
                                        "query": "{{query}}"
                                    }
                                }
                            }
                        ]
                    }
                }
            ),

            QueryFeatureExtractor(
            feature_name="category_score_all_terms",
            query={
                "match": {
                    "nomeCategoria": {"query": "{{query}} "}
                }
            },
            ),
            # QueryFeatureExtractor(
            #     feature_name="boost_dataAtualizacao",
            #     query={
            #         "script_score": {
            #             "query": {"exists": {"field": "dataAtualizacao"}},
            #             "script": {
            #                 "source": """
            #                         double maxDias = 180;
            #                         if (doc['dataAtualizacao'].size() == 0) {
            #                              return 0.0;
            #                         }
            #                         long nowMillis = new Date().getTime();
            #                         long docMillis = doc['dataAtualizacao'].value.toInstant().toEpochMilli();
            #                         double diff = (nowMillis - docMillis) / 86400000L; // divide por milissegundos em 1 dia
            #                         diff = Math.min(diff, maxDias);
            #                         return 1.0 - (diff / (double) maxDias);
            #                 """
            #             }
            #         }
            #     }
            # ),
            QueryFeatureExtractor(
                feature_name="boost_categoria",
                query={
                    "script_score": {
                        "query": {"exists": {"field": "idCategoria"}},
                        "script": {
                            "source": """
                                Map boosts = new HashMap();
                                boosts.put(145L, 1);
                                boosts.put(143L, 1);
                                boosts.put(1588L, 1);
                                long cat = doc['idCategoria'].size() > 0
                                ? doc['idCategoria'][doc['idCategoria'].size() - 1]
                                : -1L;
                                return boosts.containsKey(cat) ? boosts.get(cat) : 0;
                            """
                        }
                    }
                }
            ),
            QueryFeatureExtractor(
            feature_name="boost_popularidade",
            query={
                "script_score": {
                    "query": {"exists": {"field": "popularidade"}},
                    "script": {"source": "return Math.log(1 + doc['popularidade'].value);"},
                }
            },
            ),
            QueryFeatureExtractor(
            feature_name="boost_maisVendidos",
            query={
                "script_score": {
                    "query": {"exists": {"field": "maisVendidos"}},
                    "script": {"source": "return Math.log(1 + doc['maisVendidos'].value);"},
                }
            },
            ),
            QueryFeatureExtractor(
            feature_name="boost_percentualDesconto",
            query={
                "script_score": {
                    "query": {"exists": {"field": "percentualDesconto"}},
                    "script": {"source": "return doc['percentualDesconto'].value;"},
                }
            },
            ),
            QueryFeatureExtractor(
            feature_name="boost_classificacaoMedia",
            query={
                "script_score": {
                    "query": {"exists": {"field": "classificacaoMedia"}},
                    "script": {"source": "return doc['classificacaoMedia'].value;"},
                }
            },
            )
        ]
)
