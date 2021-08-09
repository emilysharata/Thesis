'''
Created on 1 juin 2021

@author: Sami
'''

import json
import sys
import codecs
from bs4 import BeautifulSoup
import re 
from easynmt import EasyNMT

from scipy import spatial
from googletrans import Translator

from easynmt import EasyNMT
from requests.exceptions import ReadTimeout



def readJsonFile(jsonFile, encoding="NONE") : # utf-8 #
    if(encoding=="NONE") :
        with open(jsonFile) as file :
            jsonData=json.load(file)
    else :
        #with open(jsonFile, 'r', encoding='utf-8') as file :
        with codecs.open(jsonFile, 'r', encoding=encoding, errors='ignore') as file :
            jsonData=json.load(file)
    
    return jsonData

def outputJsonData(jsonData) :
    encoding='utf8'
    jsonData_as_string=json.dumps(jsonData, indent=2, ensure_ascii=False).encode(encoding).decode()
    #sys.stdout.reconfigure(encoding=encoding)
    print(jsonData_as_string+"\n")
    

def addTranslation(alljobs, outputFile, method="google", fallback=True, maxWrite=-1, index=-1):
    if method == "google":
        from googletrans import Translator
        modelGoogle = Translator()
    if method == "easynmt" or fallback:
        from easynmt import EasyNMT
        modelNMT = EasyNMT('opus-mt')
    
    for i, entry in enumerate(alljobs[:maxWrite]) :
        entry["CLEANED_JOBS"] = textCleaner(entry["CONTENT"])
        
        # Seems to be a maximum number of characters allowed for translation
        # Split in half at a sentence boundary
        sentences = entry["CLEANED_JOBS"].split(". ")
        chunks = [". ".join(sentences)]
        entry["TRANSLATED_JOBS"] = ""
        if len(entry["CLEANED_JOBS"]) > 5000:
            mid = int(len(sentences)/2)
            chunks = [". ".join(sentences[:mid]), ". ".join(sentences[mid:])]
        for chunk in chunks:
            if method == "google":
                try:
                    translate = modelGoogle.translate(chunk, dest="en")
                    translate = translate.text
                except (TypeError, AttributeError, IndexError, ReadTimeout) as e:
                    print("Failed to translate entry", i)
                    print("Error was:", e)
                    if fallback:
                        print("Trying instead with EasyNMT")
                        translate = modelNMT.translate(chunk, target_lang="en")
                    else:
                        print("Leaving untranslated!")
                        translate = entry["CLEANED_JOBS"]
            else:
                translate = modelNMT.translate(chunk, target_lang="en")
            entry["TRANSLATED_JOBS"] += translate
        entry["TOKENIZED_JOBS"] = sentenceSplitter(entry["TRANSLATED_JOBS"])
    
    if outputFile:
        with open(outputFile, "w") as f:
            json.dump(alljobs[:maxWrite], f)
            
    return alljobs

#------------------------------- Main

# Run this if you want to do the translation, otherwise just use the translation directly
def writeTranslatedJobs(maxWrite, method="google", fallback=True, firstEntry=0, writeEvery=1000):
    input = "Data/output_V1.1.json"
    alljobs = readJsonFile(input)
    app = ".translate.json" if maxWrite < 0 else (".%i.translate.json" % maxWrite)
    if maxWrite < 0: 
        maxWrite = len(alljobs)

    translated = []
    for i in range(firstEntry, min(maxWrite, len(alljobs)), writeEvery):
        chunk = alljobs[i:i+min(writeEvery, len(alljobs)-i)]
        print(i, len(chunk))
        translated += addTranslation(chunk, input.replace(".json", "_%i%s"%(i,app)), method=method, fallback=fallback, maxWrite=maxWrite)

    with open(outputFile, "w") as f:
        json.dump(alljobs[:maxWrite], f)


def translateParagraph(chunk, method, fallback = True) :
    if method == "google":
        from googletrans import Translator
        modelGoogle = Translator()
    if method == "easynmt" or fallback:
        from easynmt import EasyNMT
        modelNMT = EasyNMT('opus-mt')

    if method == "google":
        try:
            translate = modelGoogle.translate(chunk, dest="en")
            translate = translate.text
        except (TypeError, AttributeError, IndexError, ReadTimeout) as e:
            print("Failed to translate entry")
            print("Error was:", e)
            if fallback:
                print("Trying instead with EasyNMT")
                translate = modelNMT.translate(chunk, target_lang="en")
            else:
                print("Leaving untranslated!")
                translate = chunk
    else:
        translate = modelNMT.translate(chunk, target_lang="en")
    return translate 



def main() :

    print("\n-------------------------------------------- Reading JSON files for the Joboffers Dataset --------------------------------------------\n")
    
    jsonFile_forX28Dataset="C:/Users/Sami/Documents/Sami/Teaching/2021/from 2021-02/Internships/Project of Job Offers/Dataset/X28-Data/output_V1.json"
    encodingOfJsonFile_forX28Dataset="utf-8"
    
    jsonFile_forActivitiesAndSkillsClasses="C:/Users/Sami/Documents/Sami/Teaching/2021/from 2021-02/Internships/Project of Job Offers/Dataset/Classes/Dataset_Classes.json"
    encodingOfJsonFile_forActivitiesAndSkillsClasses="NONE"
    
    print("sys.stdin.encoding = "+sys.stdin.encoding)
    
    jsonData_ofX28Dataset=readJsonFile(jsonFile_forX28Dataset, encoding=encodingOfJsonFile_forX28Dataset)
    outputJsonData(jsonData_ofX28Dataset)
    
    jsonData_ofActivitiesAndSkillsClasses=readJsonFile(jsonFile_forActivitiesAndSkillsClasses, encoding=encodingOfJsonFile_forActivitiesAndSkillsClasses)
    outputJsonData(jsonData_ofActivitiesAndSkillsClasses)
    
    print("\ntype(jsonData_ofX28Dataset)                 = "+str(type(jsonData_ofX28Dataset))+" ----> len(jsonData_ofX28Dataset) = "+str(len(jsonData_ofX28Dataset)))
    print("\ntype(jsonData_ofActivitiesAndSkillsClasses) = "+str(type(jsonData_ofActivitiesAndSkillsClasses))+" ----> len(jsonData_ofActivitiesAndSkillsClasses) = "+str(len(jsonData_ofActivitiesAndSkillsClasses)))
   
    jobOffer=jsonData_ofX28Dataset[0]
    print("\njobOffer = \n"+str(jobOffer))
    
    numberOfActivities=jsonData_ofActivitiesAndSkillsClasses["numberOfActivities"]
    activities=jsonData_ofActivitiesAndSkillsClasses["activities"]
    print("\nlen(activities) = "+str(len(activities))+" --- numberOfActivities = "+str(numberOfActivities))
    activity=activities[0]
    print("\nactivity = \n")
    outputJsonData(activity)
    
    numberOfSkills=jsonData_ofActivitiesAndSkillsClasses["numberOfSkills"]
    skills=jsonData_ofActivitiesAndSkillsClasses["skills"]
    print("\nlen(skills) = "+str(len(skills))+" --- numberOfSkills = "+str(numberOfSkills))
    skill=skills[0]
    print("\nskill = \n")
    outputJsonData(skill)

if __name__ == "__main__" :
    main()

    
def textCleaner(text) :
    soup = BeautifulSoup(text, features="html.parser")
    textNoHtml = soup.get_text()
    string_no_punct=re.sub(r"[^\w\s\.']",'',textNoHtml)
    string_no_punct = re.sub(r'\.(\w)','. \\1', string_no_punct)
    string_no_punct = re.sub(r'\n',' ', string_no_punct)
    string_no_punct = string_no_punct.replace(u'\xa0', ' ')
    return string_no_punct

def sentenceSplitter(text, nDrop=0) :
    cleanSentences = []
    for sentence in text.strip().split(".") :
        if len(sentence.split(" ")) >= nDrop:
            cleanSentences.append(sentence)
    return cleanSentences


    
