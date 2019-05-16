import kenlm
from nltk.util import everygrams
from simplifier.functions import *

class NgramModel:

  def __init__(self, lmFile, leftWind, rightWind):
    self.langModel = kenlm.LanguageModel(lmFile)
    self.left = leftWind
    self.right = rightWind

  def get_score(self, sentTokens, word, candidate):
  #         print(word , " " , candidate)

    if len( word.split() ) >1:
      sentence = " ".join(sentTokens)
      sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ',sentence )
      sentence = sentence.replace(word , "#")
      sentTokens = sentence.split()
      if "#" not in sentTokens:
        print("MOSHKLAAAAAAAAAA TNYAAA 3ND ",self.left ," MODEL")
        print(sentence)
        print(word , " " , candidate)
        index =  0
      else :
        index = sentTokens.index("#")
      sentTokens[index] =  word
    if word in sentTokens:
      index = sentTokens.index(word)
    else :
      print("MOSHKLA KBERRRRRRAAAAAAA 3ND ",self.left, " MODEL")
      print(word , "###" ,candidate, sentTokens)
      index = 0
    ## Get window words
    if index ==0:
        left = index
        right = min(len(sentTokens) , index+ self.left+self.right)
    elif index == len(sentTokens) -1 :
         right = index
         left= max(0 , index-self.right - self.left)
    else :
        left = max(0, index - self.left)
        right = min(len(sentTokens) - 1, index + self.right)
    scores = []
    tokens = sentTokens[left:right + 1]
    #print(tokens)

    sentTokens[index] = candidate  ## Put candidate in sentence to test its score
    tokens = sentTokens[left:right + 1]
    nGrams = list(everygrams(tokens))
    possibleNgrams = [n for n in nGrams if candidate in n]
    candidateScores=[]
    for n in possibleNgrams:
        #             print("N gram is " , n )
        bosFlag = (sentTokens.index(n[0]) == 0 and left == 0)
        eosFlag = (sentTokens.index(n[len(n) - 1]) == len(sentTokens) - 1 and right == len(sentTokens) - 1)
        #             print(" Eos flag ",eosFlag , " BOS flag",bosFlag)
        candidateScores.append(self.langModel.score(" ".join(n), eos=eosFlag, bos=bosFlag))
    #print("Candidate : ", candidate)
    #         print(candidateScores)
    scores.append(np.average(candidateScores))
    return np.average(candidateScores)

  def evaluate_context(self, sentence, word):
    sentTokens = sentence.split()

    if word in sentTokens:
      index = sentTokens.index(word)
    else: return False
    if index ==0:
        left = index
        right = min(len(sentTokens) , index+ self.left+self.right)
    elif index == len(sentTokens) -1 :
         right = index
         left= max(0 , index-self.right - self.left)
    else :
        left = max(0, index - self.left)
        right = min(len(sentTokens) - 1, index + self.right)

    scores = []
    tokens = sentTokens[left:right + 1]
    nGrams = list(everygrams(tokens))
    possibleNgrams = [n for n in nGrams if word in n]
    wordScores=[]
    for n in possibleNgrams:
        #             print("N gram is " , n )
        bosFlag = (sentTokens.index(n[0]) == 0 and left == 0)
        eosFlag = (sentTokens.index(n[len(n) - 1]) == len(sentTokens) - 1 and right == len(sentTokens) - 1)
        #             print(" Eos flag ",eosFlag , " BOS flag",bosFlag)
        wordScores.append(self.langModel.score(" ".join(n), eos=eosFlag, bos=bosFlag))
    #print("Candidate : ", candidate)
    #         print(candidateScores)
    scores.append(np.average(wordScores))
    return np.average(wordScores)

