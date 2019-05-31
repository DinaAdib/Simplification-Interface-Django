from nltk.stem import WordNetLemmatizer
import re
import pyphen
from collections import Counter
import numpy as np
import csv
import pandas as pd
import nltk
from collections import defaultdict
import math
import string
import re
from nltk.corpus import treebank
from nltk import Tree
from nltk.stem import PorterStemmer
from wordfreq import zipf_frequency
from stanfordcorenlp import StanfordCoreNLP
import pickle
import os
PARSER = StanfordCoreNLP('/home/dina/.local/lib/python3.5/site-packages/stanford-corenlp', lang='en', memory='4g')
lemmatizer = WordNetLemmatizer()
dictionary = pyphen.Pyphen(lang='en')

## Constants and Variables

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DIRECTORY = BASEDIR + '/simplifier/files/'
FEATURES_COUNT = 26
THRESHOLD = 5000
STEMMER = PorterStemmer()
academicWordsList = set(
    open(DIRECTORY+'academic-word-list.txt').read().split())  ## academic word list feature


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

lexicons = {}
with open(DIRECTORY+"lexicon.tsv") as f:
    for line in f:
        (key, val) = line.split()
        lexicons[key.lower()] = val


##Type-Token Ratio (TTR) is the ratio of number of word types (T) to total number word tokens in a text (N).
def lexicalDensity(tagged):
    taggedDict = defaultdict(int)
    verbDict = defaultdict(int)
    modifierVariation, adverbVariation = 0, 0
    nounVariation, adjectiveVariation, verbtypeVariation = 0, 0, 0
    verbVariation = 0
    totalwords = 0

    for word, tag in tagged:
        totalwords += 1
        general_type = getType(tag)
        taggedDict[general_type] += 1
        if general_type == "ADV": adverbVariation += 1
        if general_type == "PRN": modifierVariation += 1
        if general_type == "NN": nounVariation += 1
        if general_type == "ADJ": adjectiveVariation += 1
        if general_type == "VB":
            verbDict[tag] += 1
            verbVariation += 1
    if verbVariation > 1:
        verbtypeVariation = len(verbDict) / verbVariation

    return len(taggedDict) / totalwords, adverbVariation / totalwords, modifierVariation / totalwords, \
           nounVariation / totalwords, adjectiveVariation / totalwords, verbVariation / totalwords, verbtypeVariation


## Measure of Textual Lexical Diversity
def MTLD(tokens):
    mtldCount = 0
    defaultTTR = 0.72
    totalWords = 0  # len(tagged)
    types = []
    for t in tokens:
        totalWords += 1
        if type not in types:
            types.append(type)
        ttr = len(types) / totalWords
        if ttr <= defaultTTR:
            mtldCount += 1
            totalWords = 0
            types = []
    return mtldCount


ADJ = ["JJ", "JJR", "JJS"]
ADV = ["RB", "RBR", "RBS", "RP", "WRB"]
VB = ["MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
NN = ["NN", "NNS", "NNP", "NNPS"]
PRN = ["DT", "EX", "PDT", "PRP", "PRP$", "WDT", "WP", "WP$"]
OTHR = ["CD", "LS", "POS", "SYM", "TO", "UH", "INF", "FW"]
PUNCT = [".", ",", ":", "(", ")"]
CLAUSE = ["SBAR", "SBARQ", "SINV", "SQ"]


def getType(tag):
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
    else:
        return tag


def isClause(tag):
    if tag in CLAUSE:
        return True
    return False


def get_syntactic_features(sent):
    sent = sent[0: -1] + "."
    # print(sent)
    tree = Tree.fromstring(PARSER.parse(sent))
    dependency_tree = PARSER.dependency_parse(sent)
    syntactic_types = set([synt[0] for synt in dependency_tree[1:-1]])
    # print(syntactic_types)
    tree_height = tree[0].height()
    nsentences, nclauses, avg_len_clauses, nverb_phrases, avg_len_vphrases = 0, 0, 0, 0, 0
    nnoun_phrases, nprep_phrases, nsub_clauses, ncoordinating_conj = 0, 0, 0, 0
    for s in tree[0].subtrees():
        if s.label() == "S":  ##Sentence
            nsentences += 1
        if isClause(s.label()):
            nclauses += 1
            avg_len_clauses += len(s.flatten())
        if s.label() == "VP":
            nverb_phrases += 1
            avg_len_vphrases += len(s.flatten())
        if s.label() == "NP":
            if len(s.flatten()) > 2:
                nnoun_phrases += 1
        if s.label() == "PP":
            nprep_phrases += 1

        if s.label() == "CC":
            ncoordinating_conj += 1  ##TO DO: check whether it is cc phrase or clause

        if s.label() == "SBAR":
            nsub_clauses += 1  ##check whether it prints the actual length

    if nclauses > 0: avg_len_clauses /= 1.0 * nclauses
    if nverb_phrases > 0: avg_len_vphrases /= 1.0 * nverb_phrases

    syntactic_features = [tree_height, nsentences, nclauses, avg_len_clauses, nverb_phrases,
                          avg_len_vphrases, nnoun_phrases, nprep_phrases, nsub_clauses, ncoordinating_conj,
                          len(syntactic_types)]
    # print(syntactic_features)
    return syntactic_features


## it does all the preprocessing : don't preprocess !!!
def get_complexity_features_sentence(sent):
    sentence = sent.lower()
    sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
    sentence = " ".join(sentence.split())  ## remove numbers and punctuation
    tokens = nltk.word_tokenize(sentence)
    nwords = len(tokens)
    lemmatized_types = set()
    nsyllables, letters, avg_characters, awl_ratio, avg_word_freq = 0, 0, 0, 0, 0
    nlexicons, ncomplex_words, avg_lexicon_score = 0, 0, 0
    for token in tokens:
        letters += len(token)
        nsyllables += dictionary.inserted(token).count('-') + 1
        lemmatized_types.add(lemmatizer.lemmatize(token))
        avg_word_freq += zipf_frequency(token.lower(), 'en')
        if zipf_frequency(token.lower(), 'en') <= 4.3:  ## to be edited
            ncomplex_words += 1
        # if token in academicWordsList: awl_ratio += 1
        if token.lower() in lexicons:
            nlexicons += 1
            avg_lexicon_score += float(lexicons[token.lower()])

    if nlexicons > 0:
        avg_lexicon_score /= 1.0 * nlexicons
    if nwords == 0: return False, None
    ###  Word-Level-Features
    avg_word_freq /= 1.0 * nwords
    avg_characters = 1.0 * letters / nwords
    nsyllables /= 1.0 * nwords
    awl_ratio /= 1.0 * nwords
    ## Readability Scores
    coleman_liau = letters / nwords * 100 * 0.0588 - 1 / nwords * 100 * 0.296 - 15.8
    flesch_reading_ease = 206.835 - (1.015 * nwords) - (84.6 * nsyllables)
    flesch_grade_level = (0.39 * nwords) + (11.8 * nsyllables) - 15.59

    ## Feature: Type-Token Ratio
    ntypes = len(lemmatized_types)
    ttr_lemmatized = ntypes / nwords
    ttr_ratio = len(set(tokens)) / nwords
    root_ttr = ntypes / math.sqrt(nwords)
    uber_index_ttr = 0
    if ttr_lemmatized != 1:
        uber_index_ttr = math.pow(math.log(ntypes, 2), 2) / math.log(nwords / ntypes)
    ## Lexical Density
    tagged_words = nltk.pos_tag(tokens)
    lexical_dens, adverb_variation, modifier_variation, noun_variation, adjective_variation, verb_variation, verb_type_variation = lexicalDensity(
        tagged_words)

    mtld = MTLD(tokens)
    ## Syntactic Features
    syntactic_features = get_syntactic_features(sent)

    features = [nwords, avg_characters, awl_ratio, nsyllables,  ## traditional
                ncomplex_words, avg_word_freq, avg_lexicon_score,  ## word-level-complexity
                coleman_liau, flesch_reading_ease, flesch_grade_level,  ## readability
                ttr_lemmatized, ttr_ratio, root_ttr, uber_index_ttr,  ## ttr variations
                mtld, lexical_dens, adverb_variation, modifier_variation,  ## lexical density
                noun_variation, adjective_variation, verb_variation, verb_type_variation]
    features.extend(syntactic_features)

    return True, features


def get_complexity_scores(sent1 , sent2 , clf):
    sent1 = list(sent1.split('\n'))
    sent2 = list(sent2.split('\n'))
    sent1_scores, sent2_scores = [], []
    score1, score2 = 0,0
    for sentence in sent1:
        _, features1 = get_complexity_features_sentence(sentence)
        sent1_scores.append(clf.predict(np.array(features1).reshape(1 , -1)))
    score1 = np.average(sent1_scores)

    for sentence in sent2:
        _, features2 = get_complexity_features_sentence(sentence)
        sent2_scores.append(clf.predict(np.array(features2).reshape(1 , -1)))
    score2 = np.average(sent2_scores)
    return score1 , score2
# sent1 = ''
# with open(DIRECTORY+'newselaTestComp.txt', 'r') as f:
#     for line in f:
#         sent1 += line
#     # sent1 = sent1.replace('\n', ' ')
# f.close()

# sent1="the emancipation of women can only be completed when a fundamental transformation of living is effected ; and life-styles will change only with the fundamental transformation of all production and the establishment of a communist economy .\n the release of women can only be completed when a basic process of living is established ; and life-styles will change only with the basic process of all production and the proof of a communist economy ."
# sent2= " the release of women can only be completed when a basic process of living is established ; and life-styles will change only with the basic process of all production and the proof of a communist economy ."
# complexity_clf = load_obj(DIRECTORY + "complexity_model")
# # sent1
# complexity_score = get_complexity_scores(sent1, sent2, complexity_clf)
# print(complexity_score)