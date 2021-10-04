import nltk
import spacy
from textblob.np_extractors import FastNPExtractor


class CustomizedFastNPExtractor:
    grammar = "NP: {<DT>?<JJ|MD>*<NN|NNS>*}"

    def __init__(self, grammar=grammar):
        self._trained = False
        # self.cp = nltk.RegexpChunkParser(grammar)
        self.cp = nltk.RegexpParser(grammar)

    def train(self):
        train_data = nltk.corpus.brown.tagged_sents(categories='news')
        regexp_tagger = nltk.RegexpTagger([
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
            (r'(-|:|;)$', ':'),
            (r'\'*$', 'MD'),
            (r'(The|the|A|a|An|an)$', 'AT'),
            (r'.*able$', 'JJ'),
            (r'^[A-Z].*$', 'NNP'),
            (r'.*ness$', 'NN'),
            (r'.*ly$', 'RB'),
            (r'.*s$', 'NNS'),
            (r'.*ing$', 'VBG'),
            (r'.*ed$', 'VBD'),
            (r'.*', 'NN'),
            ])
        unigram_tagger = nltk.UnigramTagger(train_data, backoff=regexp_tagger)
        self.tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)
        self._trained = True
        return None

    def extract(self, sentence):
        if not self._trained:
            self.train()
        tokens = nltk.word_tokenize(sentence)
        # pos_tag = self.tagger.tag(tokens)
        pos_tag = nltk.pos_tag(tokens)
        parsed = self.cp.parse(pos_tag)
        print(parsed)
        # for chunk in parsed:
        #     if hasattr(chunk, "label"):
        #         print(chunk.label(), " ".join(c[0] for c in chunk))


class SpacyNPExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def extract(self, sentence):
        analyzed = self.nlp(sentence)
        noun_phrases = list(analyzed.noun_chunks)
        return noun_phrases


__all__ = [
    "FastNPExtractor",
    "CustomizedFastNPExtractor",
]
