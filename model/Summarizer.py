import Preprocess
import numpy as np
import sys


def updateIdf(n,noOfSentences,eps=0.25):
    avgIdf=0
    idf = {}
    for key in list(n.keys()):
        idf[key]=idf[key]=np.log((noOfSentences-n[key]+0.5))-np.log(n[key]+0.5)
        avgIdf+=idf[key]
    idfKeys=list(idf.keys())
    avgIdf=avgIdf/len(idfKeys)
    for key in idfKeys:
        if idf[key]<=0:
            idf[key] = eps * avgIdf
            continue


    return idf

def getScore(index1,index2,idf,avgDL,sentencesMaps,tokenizedSentences,k=1.2,b=0.75):
    score=0
    dl=len(tokenizedSentences[index2])
    for word in tokenizedSentences[index1]:
        if word not in sentencesMaps[index2]:
            continue
        den=1-b+b*(dl/avgDL)
        den=den*k
        den+=sentencesMaps[index2][word]
        score+=(idf[word]*sentencesMaps[index2][word]*(k+1))/den
    return score

def createGraph(n,noOfSentences,avgDL,sentencesMaps,tokenizedSentences):
    idf=updateIdf(n,noOfSentences)
    dag = np.zeros((noOfSentences, noOfSentences))
    for i in range(noOfSentences-1):
        for j in range(i+1,noOfSentences):
            dag[i][j] =getScore(i,j,idf,avgDL,sentencesMaps,tokenizedSentences)
            dag[j][i]=dag[i][j]
    return dag,idf

def pagerankWeighted(graph, noOfSentences, initialValue=None, damping=0.85):
    """Calculates PageRank for an undirected graph"""
    convergenceThreshold = 0.0001
    if initialValue == None: initialValue = 1.0 / noOfSentences
    scores = dict.fromkeys(range(noOfSentences), initialValue)

    iterationQuantity = 0
    for iterationNumber in range(100):
        iterationQuantity += 1
        convergenceAchieved = 0
        for i in range(noOfSentences):
            rank = 1 - damping
            for j in range(noOfSentences):
                neighborsSum=sum(graph[j])
                if neighborsSum==0:
                    continue
                rank += damping * scores[j] * graph[i][j] / neighborsSum
            if abs(scores[i] - rank) <= convergenceThreshold:
                convergenceAchieved += 1

            scores[i] = rank

        if convergenceAchieved == noOfSentences:
            break

    return scores

def summarize(ratio, txtPath):
    sentences, textLen, sentencesMaps, avgDL, n, tokenizedSentences= Preprocess.textPreprocessing(txtPath)
    dag,idf=createGraph(n,len(sentences),avgDL,sentencesMaps,tokenizedSentences)
    scores=pagerankWeighted(dag, len(sentences))
    ratio=ratio*len(sentences)
    sumSentences=[]
    i=0
    for key, value in sorted(scores.items(), key=lambda kv: kv[1],reverse=True):
        sumSentences.append(key)
        i+=1
        if i>ratio:
            break
    sumSentences.sort()
    extractiveSum=""
    for i in sumSentences:
        extractiveSum+=sentences[i]

    return extractiveSum

if __name__ == "__main__":
   
   outputFile=open(str(sys.argv[1]),'w')
   compression=str(sys.argv[2])
   inputPath=str(sys.argv[3])
   output=summarize(float(eval(compression)), str(inputPath))
   outputFile.write(output)
