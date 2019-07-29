from typing import List

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')


def make_hypernim_hierarchy(subject):
    hypernym_hierarchy: List[List[str]] = []
    while subject is not None:
        hypernym_hierarchy.append(subject.lemma_names())
        hypernyms = subject.hypernyms()
        if len(hypernyms) == 0:
            break
        subject = hypernyms[0]
    return hypernym_hierarchy


def test_hypernyms():
    cat = wordnet.synsets('cat')[0]
    assert 'feline mammal usually having thick soft fur and no ability to roar' in cat.definition()

    hypernym_hierarchy = make_hypernim_hierarchy(cat)
    assert hypernym_hierarchy == [
        ['cat', 'true_cat'],
        ['feline', 'felid'],
        ['carnivore'],
        ['placental', 'placental_mammal', 'eutherian', 'eutherian_mammal'],
        ['mammal', 'mammalian'],
        ['vertebrate', 'craniate'],
        ['chordate'],
        ['animal', 'animate_being', 'beast', 'brute', 'creature', 'fauna'],
        ['organism', 'being'],
        ['living_thing', 'animate_thing'],
        ['whole', 'unit'],
        ['object', 'physical_object'],
        ['physical_entity'],
        ['entity']
    ]


def test_common_hypernym():
    dog = wordnet.synsets('dog')[0]
    cat = wordnet.synsets('cat')[0]
    common = dog.lowest_common_hypernyms(cat)[0]
    hypernym_hierarchy = make_hypernim_hierarchy(common)
    assert hypernym_hierarchy == [
        ['carnivore'],
        ['placental', 'placental_mammal', 'eutherian', 'eutherian_mammal'],
        ['mammal', 'mammalian'],
        ['vertebrate', 'craniate'],
        ['chordate'],
        ['animal', 'animate_being', 'beast', 'brute', 'creature', 'fauna'],
        ['organism', 'being'],
        ['living_thing', 'animate_thing'],
        ['whole', 'unit'],
        ['object', 'physical_object'],
        ['physical_entity'],
        ['entity']
    ]


def test_specificity():
    assert wordnet.synsets('cat')[0].min_depth() == 13
    assert wordnet.synsets('carnivore')[0].min_depth() == 11
    assert wordnet.synsets('mammal')[0].min_depth() == 9
    assert wordnet.synsets('animal')[0].min_depth() == 6
    assert wordnet.synsets('living_thing')[0].min_depth() == 4
    assert wordnet.synsets('entity')[0].min_depth() == 0


def test_similarity():
    dog = wordnet.synsets('dog')[0]
    cat = wordnet.synsets('cat')[0]
    computer = wordnet.synsets('computer')[0]

    assert dog.path_similarity(cat) == cat.path_similarity(dog)
    assert dog.path_similarity(cat) > dog.path_similarity(computer)
