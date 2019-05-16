import io
import re
import string
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords


def readFile(fileName):
    openedFile=io.open(fileName)
    text=openedFile.read()
    openedFile.close()
    return text
def segment(text):
    sentences= sent_tokenize(text)#re.split(r'[!;:\.\?]',str(text))
    return sentences,len(text)

def stemAndStopWords(text,n,st,stops):
    sentenceWordMap={}
    word_tokens = word_tokenize(text)
    filteredSentence = []
    for w in word_tokens:
        stem=st.stem(w)
        if stem not in stops:
            filteredSentence.append(stem)
            if stem in sentenceWordMap:
                sentenceWordMap[stem]+=1
            else:
                sentenceWordMap[stem] = 1
                if stem in n:
                    n[stem] += 1
                else:
                    n[stem] = 1

    return filteredSentence,sentenceWordMap


reWhitespace = re.compile(r"(\s)+")
reNumeric = re.compile(r"[0-9]+")
reTags = re.compile(r"<([^>]+)>")
rePunct = re.compile('([%s])+' % re.escape(string.punctuation))

def tokenize(txtList):
    stops = set(stopwords.words('english'))
    st = PorterStemmer()
    n={}
    sentencesMaps=[]
    tokenizedSentences=[]
    avgDL=0
    i=0
    while i< len(txtList):
        sentence=str(txtList[i])
        sentence =reNumeric.sub(" ",sentence)
        sentence = reTags.sub(" ", sentence)
        sentence = rePunct.sub(" ", sentence)
        sentence = reWhitespace .sub(" ", sentence)
        #sentence=' '.join(sentence.split())
        filteredSentence, sentenceWordMap=stemAndStopWords(sentence, n, st, stops)
        if len(filteredSentence)<=1:
            txtList.pop(i)
            continue
        avgDL+=len(filteredSentence)
        tokenizedSentences.append(filteredSentence)
        sentencesMaps.append(sentenceWordMap)
        i+=1
    avgDL=avgDL/len(tokenizedSentences)
    return sentencesMaps,avgDL,n,tokenizedSentences

def textPreprocessing(file):
    sentences,textLen=segment(readFile(file))
    sentencesMaps, avgDL, n, tokenizedSentences=tokenize(sentences)
    return sentences,textLen,sentencesMaps, avgDL, n, tokenizedSentences

#textPreprocessing("Untitled Document.txt")


