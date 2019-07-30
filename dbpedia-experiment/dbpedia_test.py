from SPARQLWrapper import SPARQLWrapper, JSON


def test_it_works():
    sparql = SPARQLWrapper('http://dbpedia.org/sparql')
    sparql.setQuery('''
        select ?e ?type ?description
        where { 
            ?e foaf:name "Anne Hathaway"@en.
            ?e dct:description ?description.
        }
    ''')
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()
    assert result['results']['bindings'] == [
        {
            'e': {
                'type': 'uri',
                'value': 'http://dbpedia.org/resource/Anne_Hathaway'
            },
            'description': {
                'type': 'literal',
                'xml:lang': 'en',
                'value': 'American actress'
            }
        },
        {
            'e': {
                'type': 'uri',
                'value': "http://dbpedia.org/resource/Anne_Hathaway_(Shakespeare's_wife)"
            },
            'description': {
                'type': 'literal',
                'xml:lang': 'en',
                'value': 'English woman, wife of William Shakespeare'
            }
        }
    ]
