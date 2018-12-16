from typing import List
from common import NamedEntity
import nltk

nltk_data_path = './nltk-data'
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('maxent_ne_chunker', download_dir=nltk_data_path)
nltk.download('words', download_dir=nltk_data_path)

nltk.data.path.append(nltk_data_path)


def extract_named_entities(text: str) -> List[NamedEntity]:
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    chunks = nltk.chunk.ne_chunk(tagged_tokens)

    named_entities: List[NamedEntity] = []

    for chunk in chunks:
        if type(chunk) == nltk.tree.Tree:
            entity_type = chunk.label()
            entity_text = ' '.join(map(lambda text_and_tag_tuple: text_and_tag_tuple[0], chunk.leaves()))
            named_entities.append(NamedEntity(entity_type, entity_text))

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


def test_nltk():
    named_entities = extract_named_entities(text)
    for named_entity in named_entities:
        print(f'type={named_entity.type} text={named_entity.text}')
    assert len(list(filter(lambda ne: ne.type == 'ORGANIZATION' and ne.text == 'Lakeside School', named_entities))) > 0
    assert len(list(filter(lambda ne: ne.type == 'PERSON' and ne.text == 'Gates', named_entities))) > 0
