import itertools
import spacy

import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet


try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
lancaster = LancasterStemmer()
spacy_nlp = spacy.load("en_core_web_sm")


def synsets(word):
    return wordnet.synsets(word)


def all_attr_of_synsets(_query, attr_name, filter_pos=None):
    output = list(map(
        lambda _syn: getattr(_syn, attr_name, "")(),
        filter(
            lambda _syn: True if filter_pos is None else
            _syn.pos() == filter_pos,
            wordnet.synsets(_query)
        )
    ))
    return output


def stemming(word):
    stemmed = [word]
    for stemmer in (porter, lancaster):
        _stemmed = stemmer.stem(word)
        if _stemmed.lower() != word.lower() and \
                _stemmed.lower()+"e" != word.lower():
            for _suffix in ["", "e"]:
                if len(wordnet.synsets(_stemmed+_suffix)) != 0:
                    stemmed.append(_stemmed + _suffix)
    return stemmed


def get_headword(phrase):
    parsed = spacy_nlp(phrase)
    return parsed[0].head.text


def are_synonyms(word1, word2=None,
                 lemma_names_except_pos="",
                 detect_def_head_words=False,
                 ):
    if word2 is None:
        assert isinstance(word1, tuple)
        word1, word2 = word1
    # return intersect(stemming(word1), synset_by_LM(word2)) or \
    #     intersect(stemming(word2), synset_by_LM(word1))
    for _word1, _word2 in ((word1, word2), (word2, word1)):
        if detect_def_head_words:
            headwords = [
                get_headword(_definition) for _definition in
                all_attr_of_synsets(_word2, "definition", "n")[:2]
                if any(map(lambda x: _definition.startswith(x),
                           ("a ", "an ", "the")))
            ]
            if _word1 in headwords:
                return True
        _to_search = [_word2] + stemming(_word2)
        if "n" not in lemma_names_except_pos:
            _to_search += list(itertools.chain(
                *all_attr_of_synsets(_word2, "lemma_names", "n")[:2]
            ))
        if "v" not in lemma_names_except_pos:
            _to_search += list(itertools.chain(
                *all_attr_of_synsets(_word2, "lemma_names", "v")
            ))
        if _word1 in _to_search:
            return True
        # or _word1 in ". ".join(
        #     all_attr_of_synsets(_word2, "definition")
        # ):
        # if _word1 in list(map(
        #     lambda x: x[0],
        #     all_attr_of_synsets(_word2, "lemma_names", "v")
        # )):
        #     return True
    return False
