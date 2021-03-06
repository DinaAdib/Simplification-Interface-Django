# -*- coding: utf-8 -*-
"""meowLS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bc7GAqq50ZsW_8zQQ8PNje1wJoIodFcS
"""

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
import os

"""**lexicon**"""
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DIRECTORY = BASEDIR + '/simplifier/files/'
global initializer
lexicons = {}
with open(DIRECTORY + "lexicon.tsv") as f:
    for line in f:
        (key, val) = line.split()
        lexicons[key.lower()] = val

# """**Model**"""
#
# from sklearn.preprocessing import maxabs_scale
# import re
#
#################Functions######################
import pyphen
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import requests
import numpy as np
from nltk.stem.porter import *
import os  # importing os to set environment variable
from wordfreq import zipf_frequency
import sys
from sklearn.preprocessing import maxabs_scale
import pickle
from nltk import *
from pattern.text.en import referenced


######## General Functions ############
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def correct_articles(sentence):
    sentence_splitted = sentence.split()
    for i, w in enumerate(sentence_splitted):
        if w == 'a' or w == 'an':
            print(sentence_splitted[i + 1].lower()[0])
            sentence_splitted[i] = referenced(sentence_splitted[i + 1]).split()[0]
            sentence_splitted[i + 1] = referenced(sentence_splitted[i + 1]).split()[1]
            # if sentence_splitted[i + 1].lower()[0] in vowels:
            #     sentence_splitted[i] = 'an'

        # elif w == 'an':
        #     print(sentence_splitted[i + 1].lower()[0])
        #     if sentence_splitted[i + 1].lower()[0] not in vowels:
        #         sentence_splitted[i] = 'a'
    corrected = ' '.join(sentence_splitted)
    return corrected


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    if current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)
            current_chunk = []
    return continuous_chunk


#################Substitution Generation##########################
################################# SUBSTUTUTION GENERATION ###################################
from stanfordcorenlp import StanfordCoreNLP

# from functions import *

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


def get_synonyms_thesaurus(word, thesaurus):
    if word not in thesaurus:
        return False, []
    synonyms_list = thesaurus[word]
    # print(synonyms_list)
    synonyms = []
    for line in synonyms_list:
        synonyms.extend(line.split("|")[1:-1])
    print(synonyms)
    return True, synonyms


def get_candidates(complex_word):
    candidates, most_frequent = [], []

    global initializer
    ppdb_candidates, thesaurus_candidates, wordnet_candidates = [], [], []
    complex_tag_specific = NLP.pos_tag(complex_word)[0][1]
    complex_tag = get_type(complex_tag_specific)
    ppdb_substitutes = initializer.ppdb_substitutes
    if complex_word in ppdb_substitutes.keys():
        ppdb_candidates = ppdb_substitutes[complex_word]
        candidates.extend(ppdb_candidates)


    mythesaurus = initializer.mythesaurus
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
            final_candidates.add(candidate)
    elif generic_tag == "ADJ":  ## Adjectives
        for candidate in candidates:
            candidate_tag = NLP.pos_tag(candidate)[0][1]
            if specific_tag == "JJR" and candidate_tag != "JJR":
                candidate = comparative(candidate)
            elif specific_tag == "JJS" and candidate_tag != "JJS":
                candidate = superlative(candidate)

            final_candidates.add(candidate)
    elif generic_tag == "VB":  ## Verbs
        complex_tense = tenses(complex_word)
        if (len(complex_tense)) < 1: return candidates

        for candidate in candidates:
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


FEATURES_COUNT = 8

def get_similarity_scores(candidates, complex_word):
    global initializer
    myword2vec = initializer.word2vec
    complex_vec = myword2vec.get_vector(complex_word)
    scores = []
    for candidate in candidates:
        candidate_vec = myword2vec.get_vector(candidate)
        scores.append(myword2vec.get_cosine_similarity([candidate_vec], [complex_vec]))
    return maxabs_scale(scores)


def get_ngram_scores(candidates, complex_word, sentence, model):
    scores = []
    sentence = sentence.lower()
    complex_word = re.sub(r'[^a-zA-Z0-9\s]', ' ', complex_word)
    for candidate in candidates:
        candidate = re.sub(r'[^a-zA-Z0-9\s]', ' ', candidate)
        scores.append(model.get_score(sentence.split(), complex_word.lower(), candidate.lower()))

    return maxabs_scale(scores)


def get_lexicon_scores(candidates):
    scores = []
    for candidate in candidates:
        if candidate.lower() in lexicons:
            scores.append(float(lexicons[candidate.lower()]))
        else:
            scores.append(2.5)  ## average word
    return maxabs_scale(scores)


def get_syllable_counts(candidates):
    scores = []
    global initializer
    syllable_dict = initializer.syllable_dict

    for candidate in candidates:
        scores.append(syllable_dict.inserted(candidate).count('-') + 1)
    return maxabs_scale(scores)


def get_character_counts(candidates):
    scores = []
    for candidate in candidates:
        scores.append(len(candidate))
    return maxabs_scale(scores)


def get_frequencies(candidates):
    scores = []
    global initializer
    wiki_frequency = initializer.wiki_frequency
    for candidate in candidates:
        scores.append(wiki_frequency.get_feature(candidate.lower()))
    return maxabs_scale(scores)


def get_wiki_frequencies(candidates):
    scores = []
    for candidate in candidates:
        scores.append(zipf_frequency(candidate.lower(), 'en'))
    return maxabs_scale(scores)


def get_features(filename, train=True):
    all_features = np.zeros(FEATURES_COUNT)
    all_ranks = []
    all_lines = []
    all_words = []

    global initializer
    substitutions_db = initializer.substitutions_db
    fivegram_model = initializer.fivegram_model
    threegram_model = initializer.threegram_model
    with open(DIRECTORY + filename) as file:
        corpus = file.read()
        lines = corpus.split("\n")
        increment = 0
        for line in lines:
            try:
                increment += 1
                ### Processing line
                tokens = line.strip().split('\t')
                sentence = tokens[0].strip()
                complex_word = tokens[1].strip()
                # print(complex_word)
                ranks = [int(token.strip().split(':')[0]) for token in tokens[3:]]
                if train:
                    candidates = [token.strip().split(':')[1] for token in tokens[3:]]
                else:
                    candidates = get_candidates(complex_word)

                ### Extracting Features
                cosine_similarities = get_similarity_scores(candidates, complex_word)
                fivegram_scores = get_ngram_scores(candidates, complex_word, sentence, fivegram_model)
                threegram_scores = get_ngram_scores(candidates, complex_word, sentence, threegram_model)
                lexicon_scores = get_lexicon_scores(candidates)
                syllables = get_syllable_counts(candidates)
                characters = get_character_counts(candidates)
                frequencies = get_frequencies(candidates)
                wiki_frequencies = get_wiki_frequencies(candidates)
            except:
                print("************** BIG ERROR OCCURED")
                continue

            ## No error so we will append
            for i in range(len(candidates)):
                all_lines.append(sentence)
                all_words.append(candidates[i])
            features = np.column_stack((cosine_similarities, fivegram_scores, threegram_scores, lexicon_scores,
                                        syllables, characters, frequencies, wiki_frequencies))
            all_ranks.extend(ranks)
            all_features = np.vstack((all_features, features))

            if increment > 40000:
                increment = 0
                np.save(DIRECTORY + "_meow_ftrs" + str(increment), all_features[1:len(all_features)])
                np.save(DIRECTORY + "_meow_ranks" + str(increment), all_ranks)
                np.save(DIRECTORY + "_meow_lines" + str(increment), all_lines)
                np.save(DIRECTORY + "_meow_words" + str(increment), all_words)

        return all_features[1:len(all_features)], all_ranks, all_lines, all_words


"""**Classifier**"""

"""# Full Model"""


def get_line_features(line):
    global initializer
    substitutions_db = initializer.substitutions_db
    fivegram_model = initializer.fivegram_model
    threegram_model = initializer.threegram_model
    try:
        # print(line)

        ### Processing line
        tokens = line.strip().split('\t')
        sentence = tokens[0].strip()
        complex_word = tokens[1].strip()
        # print(complex_word)
        ranks = [int(token.strip().split(':')[0]) for token in tokens[3:]]
        candidates = [token.strip().split(':')[1] for token in tokens[3:]]
        our_candidates = get_candidates(complex_word)
        # print(candidates)

        ### Extracting Features
        cosine_similarities = get_similarity_scores(candidates, complex_word)
        fivegram_scores = get_ngram_scores(candidates, complex_word, sentence, fivegram_model)
        threegram_scores = get_ngram_scores(candidates, complex_word, sentence, threegram_model)
        lexicon_scores = get_lexicon_scores(candidates)
        syllables = get_syllable_counts(candidates)
        characters = get_character_counts(candidates)
        frequencies = get_frequencies(candidates)
        wiki_frequencies = get_wiki_frequencies(candidates)
    except:
        print("************** BIG ERROR OCCURED")
        return False, None, None, None, None, None

    ## No error
    features = np.column_stack((cosine_similarities, fivegram_scores, threegram_scores, lexicon_scores, syllables,
                                characters, frequencies, wiki_frequencies))

    return True, features, our_candidates, candidates, ranks, complex_word


def preprocess_line(line):
    try:
        tokens = line.strip().split('\t')
        sentence = tokens[0].strip()
        sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
        complex_word = tokens[1].strip()
        # print(complex_word)
        ranks = [int(token.strip().split(':')[0]) for token in tokens[3:]]
        candidates = [token.strip().split(':')[1] for token in tokens[3:]]
    except:
        print("SOMETHING WRING ")
        return None, None, None, None
    return sentence, candidates, ranks, complex_word


def preprocess_interface_line(line):
    try:
        # sentence = line.strip().split('')
        # print("SENTENCE ",sentence)
        # sentence = tokens[0].strip()
        sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', line)
        complex_word = ""
        # print(complex_word)
        ranks = []
        candidates = []
    except:
        print("SOMETHING WRING ")
        return None, None, None, None
    return sentence, candidates, ranks, complex_word


def get_candidates_features(candidates, sentence, complex_word):
    global initializer
    fivegram_model = initializer.fivegram_model
    threegram_model = initializer.threegram_model
    try:
        cosine_similarities = get_similarity_scores(candidates, complex_word)
        fivegram_scores = get_ngram_scores(candidates, complex_word, sentence, fivegram_model)
        threegram_scores = get_ngram_scores(candidates, complex_word, sentence, threegram_model)
        lexicon_scores = get_lexicon_scores(candidates)
        syllables = get_syllable_counts(candidates)
        characters = get_character_counts(candidates)
        frequencies = get_frequencies(candidates)
        wiki_frequencies = get_wiki_frequencies(candidates)
    except:
        print("************** BIG ERROR OCCURED")
        return False, None,

    features = np.column_stack((cosine_similarities, fivegram_scores, threegram_scores, lexicon_scores, syllables,
                                characters, frequencies, wiki_frequencies))

    return True, features

total_count = 0
total_words_count = 0
ratio = 0
pos_ratio = 0
hit = 0
pos_hit = 0
all_ratios = []


def rank(input_path, output_path,  initializer_x):
    global initializer
    initializer = initializer_x
    subs_rank_nnclf = initializer.subs_rank_nnclf

    #########################################################
    substitutions_db = initializer.substitutions_db
    fivegram_model = initializer.fivegram_model
    # word2vec = initializer.word2vec
    print("Ranking")
    word_hit = 0
    test_lines = open(input_path).readlines()
    with open(BASEDIR + output_path, "w") as f:
        for line in test_lines:
            sentence, candidates, ranks, complex_word = preprocess_interface_line(line)
            # print(sentence)
            print(complex_word)

            simplified = line
            if sentence is not None and sentence != "":
                named_entities = get_continuous_chunks(sentence)
                for w in sentence.split():
                    if (w[0].isupper() == False and w != (sentence.split())[0]) and (w not in named_entities) and (
                            (w in lexicons and float(lexicons[w]) > 3.0) or zipf_frequency(w, 'en') <= 4.2):

                        complex_lm_score = fivegram_model.evaluate_context(sentence, w)
                        word_hit += 1

                        candidates = get_candidates(w)

                        candidates_pos = list(convert_postag(w, candidates))
                        for c in candidates_pos:
                            if zipf_frequency(c, 'en') < zipf_frequency(w, 'en'):
                                candidates_pos.remove(c)
                        print("Final candidates ", candidates_pos)
                        indicator, candidates_features = get_candidates_features(candidates_pos, sentence, w)

                        if indicator == True:
                            candidate_predictions = subs_rank_nnclf.predict(candidates_features)
                            #         best_candidates =candidates_pos[np.argmin(candidate_predictions)]  B3DEEEN
                            #         print(best_candidates)

                            print("Best ranks ", candidate_predictions)
                            min_frequency = 10
                            best_candidate = w
                            min_context = complex_lm_score
                            for cindex, candidate in enumerate(candidates_pos):
                                if candidate_predictions[cindex] == min(candidate_predictions):
                                    freq_diff = zipf_frequency(lemmatizer.lemmatize(candidate), 'en') - zipf_frequency(
                                        lemmatizer.lemmatize(w), 'en')
                                    word_lm_score = fivegram_model.evaluate_context(sentence.replace(w, candidate),
                                                                                    candidate)
                                    if freq_diff > 0 and freq_diff < min_frequency:# and word_lm_score > min_context:
                                        min_frequency = freq_diff
                                        best_candidate = candidate
                            simplified = simplified.replace(w, best_candidate, 1)
                simplified = correct_articles(simplified)
                # print(" sentence simplified: " , sentence.replace(w, best_of_best ) )
                # f.write("<original> ")
                # f.write(original)
                # f.write("\n<simplified> ")
                f.write(simplified)
                f.write("\n")
    f.close()

if __name__ == "__main__":
    # outputFile = open(str(sys.argv[1]), 'w+')
    inputPath = str(sys.argv[2])
    # inputPath = 'files/BenchLS.test.txt'
    rank(str(inputPath))
