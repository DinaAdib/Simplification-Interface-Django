from simplifier.functions import *

class WikiFrequency:
    def __init__(self):
        self.wiki_frequencies = load_obj(DIRECTORY+"wikiFreq")

    def get_feature(self, phrase):
        phrase = phrase.lower()
        if phrase in self.wiki_frequencies:
            return self.wiki_frequencies[phrase]
        return 0