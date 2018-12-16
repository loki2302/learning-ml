from invoke import task


@task
def test(c):
    """
    Run all tests
    """
    c.run('pytest')

@task
def nltktest(c):
    """
    Run nltk_test.py
    """
    c.run('pytest nltk_test.py')

@task
def spacytest(c):
    """
    Run spacy_test.py
    """
    c.run('pytest spacy_test.py')

@task
def shelltest(c):
    c.run('ls -la')
    print(c.pipiska)
