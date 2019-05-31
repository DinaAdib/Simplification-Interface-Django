################################# SUBSTUTUTION GENERATION ###################################
from stanfordcorenlp import StanfordCoreNLP
from simplify.functions import *

NLP = StanfordCoreNLP('/home/dina/.local/lib/python3.5/site-packages/stanford-corenlp', lang='en', memory='4g')

ADJ = ["JJ", "JJR", "JJS"]
ADV = ["RB", "RBR", "RBS", "RP", "WRB"]
VB = ["MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
NN = ["NN", "NNS", "NNP", "NNPS"]
PRN = ["PDT", "PRP", "PRP$", "WP", "WP$"]
DT = ["DT", "EX", "PDT", "WDT"]
OTHR = ["CD", "LS", "POS", "SYM", "TO", "UH", "INF", "FW"]
PUNCT = [".", ",", ":", "(", ")"]
CLAUSE = ["SBAR", "SBARQ", "SINV", "SQ"]


def get_type(tag):
    if tag in ADJ:
        return "ADJ"
    elif tag in ADV:
        return "ADV"
    elif tag in VB:
        return "VB"
    elif tag in NN:
        return "NN"
    elif tag in PRN:
        return "PRN"
    elif tag in OTHR:
        return "OTHR"
    elif tag in PUNCT:
        return "PUNCT"
    elif tag in CLAUSE:
        return "CONJ"
    elif tag in DT:
        return "DT"
    else:
        return tag


def map_postag(tag):
    if tag == "ADJ": return "a"
    if tag == "NN": return "n"
    if tag == "ADV": return "r"
    if tag == "VB": return "v"


def get_synonyms_api(word, tag=None):
    # api-endpoint
    url = "https://api.datamuse.com/words?ml=" + word
    # sending get request and saving the response as response object
    response = requests.get(url=url).json()
    synonyms = [line['word'] for line in response[0:6]]
    return synonyms


def get_synonyms_wordnet(word, pos):
    synset = wn.synsets(word, pos)
    #     print(synset)
    # return synonym lemmas in no particular order
    return [lemma.name() for s in synset for lemma in s.lemmas()]


def get_synonyms_thesaurus(word , thesaurus ):
  if word not in thesaurus:
      return False, []
  synonyms_list = thesaurus[word]
  #print(synonyms_list)
  synonyms=[]
  for line in synonyms_list:
    synonyms.extend(line.split("|")[1:-1])
  print(synonyms)
  return True, synonyms

ppdb_substitutes = dict(np.load(DIRECTORY + 'ppdb_candidates_2.7_4.npy', allow_pickle=True).item())
mythesaurus = load_obj(DIRECTORY + "/mythesaurus")

def get_candidates(complex_word, db):
    print("received ", complex_word)
    candidates, most_frequent = [], []

    filtered_candidates = []
    ppdb_candidates, thesaurus_candidates, wordnet_candidates = [], [], []
    complex_tag_specific = NLP.pos_tag(complex_word)[0][1]
    complex_tag = get_type(complex_tag_specific)

    if complex_word in ppdb_substitutes.keys():
        ppdb_candidates = ppdb_substitutes[complex_word]
        candidates.extend(ppdb_candidates)
        print("ppdb subs ", ppdb_substitutes[complex_word])

    # if len(candidates) < 6:
    #     wordnet_tag = map_postag(complex_tag)
    #     # print("complex tag ", complex_tag)
    #     wordnet_candidates = get_synonyms_wordnet(complex_word, wordnet_tag)
    #     print("wordnet subs ", wordnet_candidates)
    #     candidates.extend(wordnet_candidates)

    if len(candidates) < 6:
        # get synonyms from thesaurus
        found, thesaurus_candidates = get_synonyms_thesaurus(complex_word, mythesaurus)
        print("Thesaurus subs ", thesaurus_candidates)
        if found == True:
            candidates.extend(thesaurus_candidates)
    candidates = candidates[:min(6,len(candidates))]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    top_candidates = set()
    for candidate in candidates:
        candidate_tag = get_type(NLP.pos_tag(candidate)[0][1])
        # print ( "in function::" ,NLP.pos_tag( candidate ))
        if stemmer.stem(candidate) != stemmer.stem(complex_word) and lemmatizer.lemmatize(
                candidate) != lemmatizer.lemmatize(complex_word):
            top_candidates.add(candidate)
    return top_candidates




# word="circumstance"
substitutions_db = load_obj(DIRECTORY + "substitutions")

from pattern.text.en import pluralize, singularize, comparative, superlative
from pattern.text.en import conjugate
from pattern.text.en import tenses, INFINITIVE, PRESENT, PAST, FUTURE


def convert_postag(complex_word, candidates):
    specific_tag = NLP.pos_tag(complex_word)[0][1]
    generic_tag = get_type(specific_tag)
    # print(generic_tag)
    final_candidates = set()
    if generic_tag == "NN":  ### Nouns
        # print(generic_tag)
        for candidate in candidates:
            candidate_tag = NLP.pos_tag(candidate)[0][1]
            if specific_tag == "NNS" and candidate_tag != "NNS":
                candidate = pluralize(candidate)
                # print("pluraaal  ", candidate)
            elif specific_tag == "NN" and candidate_tag == "NNS":
                candidate = singularize(candidate)
                # print("singulaaar" , candidate)
            # print("wwilll add")
            final_candidates.add(candidate)
    elif generic_tag == "ADJ":  ## Adjectives
        for candidate in candidates:
            candidate_tag = NLP.pos_tag(candidate)[0][1]
            if specific_tag == "JJR" and candidate_tag != "JJR":
                candidate = comparative(candidate)
                # print(candidate , "jjr")
            elif specific_tag == "JJS" and candidate_tag != "JJS":
                # print(candidate , "jjs")
                candidate = superlative(candidate)
            # print(candidate , "added")
            final_candidates.add(candidate)
    elif generic_tag == "VB":  ## Verbs
        complex_tense = tenses(complex_word)
        if (len(complex_tense)) < 1: return candidates

        for candidate in candidates:
            # print("my tense" ,  complex_tense.upper()  ," candidate " , candidate , " ", tenses(candidate)[0][0] )
            if len(tenses(candidate)) > 0 and tenses(candidate)[0][0] != complex_tense:
                if complex_tense == "past":
                    candidate = conjugate(candidate, tense=PAST)
                elif complex_tense == "present":
                    candidate = conjugate(candidate, tense=PRESENT)
                elif complex_tense == "future":
                    candidate = conjugate(candidate, tense=FUTURE)
                elif complex_tense == "infinitive":
                    candidate = conjugate(candidate, tense=INFINITIVE)
            final_candidates.add(candidate)
    else:
        final_candidates = candidates

    return final_candidates