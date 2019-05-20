# -*- UTF-8 -*-
from elasticsearch import Elasticsearch


def search(query="", size=200):
    # replace your url
    url = "http://10.1.1.9:9266"
    es = Elasticsearch([url])
    doc = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["question.Title", "question.Body"]
            }
        }
    }
    doc2 = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "question.Title": "How to use java"
                        }
                    },
                    {
                        "match": {
                            "question.Body": "How to use java"
                        }
                    }
                ]
            }
        }
    }

    results = es.search(index="sof_raw", doc_type="posts", body=doc, size=size)
    return_list = []
    for res in results['hits']['hits']:
        return_list.append(res['_source'])

    return return_list
