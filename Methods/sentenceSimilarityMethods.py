'''
Created on 1 juin 2021

@author: Sami
'''

from sent2vec.vectorizer import Vectorizer
from scipy import spatial

from easynmt import EasyNMT

class SentenceSimilarity_abstract() :
    # this function computes the similarity between two sentences, the more similar the two snetends are the lower the 
    # computed score is
    def compute_SentenceToSentence_similarity(self, sentence1, sentence2):
        pass

class SentenceSimilarity_BERT(SentenceSimilarity_abstract) :
    def __init__(self):
        self.vectorizer = Vectorizer()
    
    # this function computes the similarity between two sentences, the more similar the two snetends are the lower the 
    # computed score is
    def compute_SentenceToSentence_similarity(self, sentenceA, sentenceB):
        sentences=[sentenceA, sentenceB]
        
        self.vectorizer.bert(sentences)
        vectors = self.vectorizer.vectors
        
        embeddingOf_sentenceA=vectors[0]
        embeddingOf_sentenceB=vectors[1]
        
        distance=spatial.distance.cosine(embeddingOf_sentenceA, embeddingOf_sentenceB)
        
        return distance

class SentenceSimilarity_translationBased(SentenceSimilarity_abstract) :
    def __init__(self):
        self.vectorizer = Vectorizer()
        self.translationModel=EasyNMT('opus-mt')
        self.targetLanguage="en"
    
    # this function computes the similarity between two sentences, the more similar the two snetends are the lower the 
    # computed score is
    def compute_SentenceToSentence_similarity(self, sentenceA, sentenceB):
        
        sourceLanguageA=self.translationModel.language_detection(sentenceA)
        translationsA=self.translationModel.translate([sentenceA], source_lang=sourceLanguageA, target_lang=self.targetLanguage)
        
        sourceLanguageB=self.translationModel.language_detection(sentenceB)
        translationsB=self.translationModel.translate([sentenceB], source_lang=sourceLanguageB, target_lang=self.targetLanguage)
        
        sentences=[translationsA[0], translationsB[0]]
        
        self.vectorizer.bert(sentences)
        vectors = self.vectorizer.vectors
        
        embeddingOf_sentenceA=vectors[0]
        embeddingOf_sentenceB=vectors[1]
        
        print("\nsentenceA \""+sentenceA+"\" --- sourceLanguageA="+sourceLanguageA+" --- translation = "+translationsA[0])
        print("sentenceB \""+sentenceB+"\" --- sourceLanguageB="+sourceLanguageB+" --- translation = "+translationsB[0])
        
        distance=spatial.distance.cosine(embeddingOf_sentenceA, embeddingOf_sentenceB)
        
        return distance

def main() :
    print("\n-------------------------------------------------- testing some sentence-similarity methods --------------------------------------------------\n")
    
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    sentenceSimilarity=SentenceSimilarity_BERT()
    
    sentences=["Hello dear friend", "Good morning Jessica", "Good morning John", "I will buy a car"]
    
    sentence=sentences[0]
    print("sentence = "+sentence)
    
    for j in range(0, len(sentences)) :
        sentenceToCompare=sentences[j]
        print("\tsentenceToCompare = "+sentenceToCompare)
        similarityScore=sentenceSimilarity.compute_SentenceToSentence_similarity(sentence, sentenceToCompare)
        print("\t\tsimilarity (using "+type(sentenceSimilarity).__name__+") = "+str(similarityScore))
    
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    
    print("\n-----------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------")
    
    sentenceSimilarity=SentenceSimilarity_translationBased()
    
    sentences=["Hello dear friend", "Bonjour cher ami", "Good morning John", "Bonjour John", "I will buy a car", "Je vais acheter une voiture"]
    sentence=sentences[0]
    print("sentence = "+sentence)
    
    for j in range(0, len(sentences)) :
        sentenceToCompare=sentences[j]
        print("\tsentenceToCompare = "+sentenceToCompare)
        similarityScore=sentenceSimilarity.compute_SentenceToSentence_similarity(sentence, sentenceToCompare)
        print("\t\tsimilarity (using "+type(sentenceSimilarity).__name__+") = "+str(similarityScore))

if __name__=="__main__" :
    main()
