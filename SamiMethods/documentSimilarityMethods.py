'''
Created on 1 juin 2021

@author: Sami
'''

from sentenceSimilarityMethods import SentenceSimilarity_BERT,\
    SentenceSimilarity_translationBased

class DocumentSimilarity_abstract() :
    # this function computes the similarity between two sentences, the more similar the two snetends are the lower the 
    # computed score is
    def compute_DocumentToDocument_similarity(self, documentA, documentB):
        pass

class DocumentSimilarity_max() :
    def __init__(self, sentenceSimilarityMethod):
        self.sentenceSimilarityMethod=sentenceSimilarityMethod
    
    # this function computes the similarity between two sentences, the more similar the two snetends are the lower the 
    # computed score is
    def compute_DocumentToDocument_similarity(self, documentA, documentB):
        minDistance=float('inf')
        
        for sentenceInA in documentA :
            for sentenceInB in documentB :
                distance=self.sentenceSimilarityMethod.compute_SentenceToSentence_similarity(sentenceInA, sentenceInB)
                minDistance=min(minDistance, distance)
        
        return minDistance

def main() :
    print("\n-------------------------------------------------- testing some document-similarity methods --------------------------------------------------\n")
    
    sentenceSimilarityMethod=SentenceSimilarity_BERT()
    documentSimilarityMethod=DocumentSimilarity_max(sentenceSimilarityMethod)
    
    documents=[
        ["Hello dear friend"],
        ["Bonjour cher ami", "hola amigo"],
        ["I will buy a car", "Which brand is it"]
    ]
    
    document=documents[0]
    print("document = "+str(document))
    
    for j in range(0, len(documents)) :
        documentToCompare=documents[j]
        print("\tdocumentToCompare = "+str(documentToCompare))
        similarityScore=documentSimilarityMethod.compute_DocumentToDocument_similarity(document, documentToCompare)
        print("\t\tsimilarity (using "+type(documentSimilarityMethod).__name__+" and "+type(sentenceSimilarityMethod).__name__+") = "+str(similarityScore))
    
    print("\n----------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------")
    sentenceSimilarityMethod=SentenceSimilarity_translationBased()
    documentSimilarityMethod=DocumentSimilarity_max(sentenceSimilarityMethod)

    print("document = "+str(document))
    
    for j in range(0, len(documents)) :
        documentToCompare=documents[j]
        print("\tdocumentToCompare = "+str(documentToCompare))
        similarityScore=documentSimilarityMethod.compute_DocumentToDocument_similarity(document, documentToCompare)
        print("\t\tsimilarity (using "+type(documentSimilarityMethod).__name__+" and "+type(sentenceSimilarityMethod).__name__+") = "+str(similarityScore))
    
    

if __name__=="__main__" :
    main()
