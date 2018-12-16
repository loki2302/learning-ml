from typing import List
import spacy
import os
from common import NamedEntity

model_path = 'spacy-data/en_core_web_sm/en_core_web_sm-2.0.0'
if not os.path.isdir(model_path):
    spacy.cli.download('en_core_web_sm-2.0.0', True, '--target', 'spacy-data')

nlp = spacy.load(model_path)


def extract_named_entities(text: str) -> List[NamedEntity]:
    doc = nlp(text)
    named_entities: List[NamedEntity] = []
    for entity in doc.ents:
        named_entities.append(NamedEntity(entity.label_, entity.text))
    return named_entities


text = '''
At 13, he enrolled in the Lakeside School, a private preparatory school and wrote his first software 
program.When Gates was in the eighth grade, the Mothers' Club at the school used proceeds from Lakeside 
School's rummage sale to buy a Teletype Model 33 ASR terminal and a block of computer time on a General 
Electric (GE) computer for the school's students. Gates took an interest in programming the GE system 
in BASIC, and was excused from math classes to pursue his interest. He wrote his first computer program 
on this machine: an implementation of tic-tac-toe that allowed users to play games against the computer. 
Gates was fascinated by the machine and how it would always execute software code perfectly. When he 
reflected back on that moment, he said, "There was just something neat about the machine." After the 
Mothers Club donation was exhausted, he and other students sought time on systems including DEC PDP 
minicomputers. One of these systems was a PDP-10 belonging to Computer Center Corporation (CCC), which 
banned four Lakeside students – Gates, Paul Allen, Ric Weiland, and Kent Evans – for the summer after it 
caught them exploiting bugs in the operating system to obtain free computer time.
'''.replace('\n', '')


def test_spacy():
    named_entities = extract_named_entities(text)
    for named_entity in named_entities:
        print(f'type={named_entity.type} text={named_entity.text}')
    assert len(list(filter(lambda ne: ne.type == 'ORG' and ne.text == 'the Lakeside School', named_entities))) > 0
    assert len(list(filter(lambda ne: ne.type == 'PERSON' and ne.text == 'Gates', named_entities))) > 0
