from Features.WikiFrequency import *
from Features.Ngram import *
from Features.Lexicon import *
dic = pyphen.Pyphen(lang='en')


substitutions_db = load_obj(DIRECTORY+"substitutions")
fivegram_model = NgramModel(DIRECTORY+"newsela.lm", 2, 2)
threegram_model = NgramModel(DIRECTORY +"newsela.lm" , 1 ,1)
syllable_dict=pyphen.Pyphen(lang='en')
wiki_frequency = WikiFrequency()


########## Ranking Functions ############
def get_similarity_scores(candidates):
  scores = []
  for candidate in candidates :
      scores.append(0.0)
  return maxabs_scale(scores)


def get_ngram_scores(candidates, complex_word, sentence, model):
    scores = []
    sentence = sentence.lower()
    complex_word = re.sub(r'[^a-zA-Z0-9\s]', ' ', complex_word)
    for candidate in candidates:
        candidate = re.sub(r'[^a-zA-Z0-9\s]', ' ', candidate)
        scores.append(model.get_score(sentence.split(), complex_word.lower(), candidate.lower()))

    # print(scores)
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
    with open(DIRECTORY + filename) as file:
        corpus = file.read()
        lines = corpus.split("\n")
        increment = 0
        for line in lines:
            try:
                increment += 1
                # print(line)
                ### Processing line
                tokens = line.strip().split('\t')
                sentence = tokens[0].strip()
                complex_word = tokens[1].strip()
                # print(complex_word)
                ranks = [int(token.strip().split(':')[0]) for token in tokens[3:]]
                if train:
                    candidates = [token.strip().split(':')[1] for token in tokens[3:]]
                else:
                    candidates = get_candidates(complex_word, substitutions_db)
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
                print("lines", len(all_lines))
                print("features: ", all_features.shape)
                print("finisheeeedd ", increment)
                increment = 0
                np.save(DIRECTORY + "_meow_ftrs" + str(increment), all_features[1:len(all_features)])
                np.save(DIRECTORY + "_meow_ranks" + str(increment), all_ranks)
                np.save(DIRECTORY + "_meow_lines" + str(increment), all_lines)
                np.save(DIRECTORY + "_meow_words" + str(increment), all_words)

        print("finallly ", np.array(all_features).shape)
        return all_features[1:len(all_features)], all_ranks, all_lines, all_words
