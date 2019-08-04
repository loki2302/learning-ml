from SPARQLWrapper import SPARQLWrapper, JSON


def test_it_works():
    # https://www.wikidata.org/wiki/Property:P463 - "member of"
    # https://www.wikidata.org/wiki/Q458 - "European Union"

    sparql = SPARQLWrapper('https://query.wikidata.org/bigdata/namespace/wdq/sparql')
    sparql.setQuery('''
        SELECT ?country ?countryLabel WHERE {
            ?country wdt:P463 wd:Q458.
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
    ''')
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()
    assert set([x['countryLabel']['value'] for x in result['results']['bindings']]) == set([
        'Ireland', 'Hungary', 'Spain', 'Belgium', 'Luxembourg', 'Finland', 'Sweden', 'Denmark', 'Poland',
        'Lithuania', 'Italy', 'Austria', 'Greece', 'Portugal', 'Netherlands', 'France', 'United Kingdom',
        'Germany', 'Estonia', 'Latvia', 'Czech Republic', 'Slovakia', 'Slovenia', 'Romania', 'Bulgaria',
        'Croatia', 'Cyprus', 'Malta', 'Kingdom of the Netherlands'
    ])
