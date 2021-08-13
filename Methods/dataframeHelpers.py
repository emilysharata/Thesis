import tokenizers
from . import JobOffers
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import os
import logging 
from scipy import spatial
logging.getLogger("tensorflow").setLevel(logging.ERROR)
basefolder = os.path.expanduser("~/EmilyThesis/Thesis")
if not os.path.isdir(basefolder):
    basefolder = os.path.expanduser("~/Desktop/Thesis")

def matchToJob(row, jobBoundaries, allEmbeddings):
    first = jobBoundaries[row.name-1] if row.name != 0 else 0
    last = jobBoundaries[row.name]
    return allEmbeddings[first:last]

def buildActivitiesDF():
    activitiesDf = pd.read_csv(os.path.expanduser(f"{basefolder}/Data/Activities.csv"), error_bad_lines=False)
    activitiesDf = activitiesDf.rename(columns={'O*Net\nMétiers verts' : "ACTIVITY_ID", 'O"Net: Détails des activités pour les métiers verts' : "ACTIVITY_CONTENT"})
    slimmedActsDf = activitiesDf[["ACTIVITY_ID", "ACTIVITY_CONTENT"]][~activitiesDf["ACTIVITY_CONTENT"].isna()]
    slimmedActsDf.loc[slimmedActsDf["ACTIVITY_ID"].isna(), 'ACTIVITY_ID'] = slimmedActsDf[slimmedActsDf["ACTIVITY_ID"].isna()].index
    return slimmedActsDf

def buildActivitiesDFV26():
    slimmedActsDf = pd.read_csv(os.path.expanduser(f"{basefolder}/Data/ActivitiesV26.csv"), sep=";", error_bad_lines=False)
    return slimmedActsDf


def getEmbeddings(actsDf, jobsDf):
    jobsSentences = np.array([y for x in jobsDf["JOB_CONTENT"] for y in x])
    activitiesSentences = actsDf["ACTIVITY_CONTENT"].to_numpy()
    allSentences = np.concatenate((jobsSentences, activitiesSentences))
    allEmbeddings = embed(allSentences)
    return allEmbeddings

def getEmbeddingsWSkills(skillsDf, jobsDf):
    jobsSentences = np.array([y for x in jobsDf["JOB_CONTENT"] for y in x])
    skillsSentences = skillsDf["SKILL_CONTENT"].to_numpy()
    allSentences = np.concatenate((jobsSentences, skillsSentences))
    allEmbeddings = embed(allSentences)
    return allEmbeddings


def addEmbeddingsToJobs(jobsDf, embeddings):
    jobBoundaries = np.cumsum([len(x) for x in jobsDf["JOB_CONTENT"]])
    jobsDf["JOB_SCORES"] = jobsDf.apply(matchToJob, args=[jobBoundaries, embeddings], axis=1)
    return jobsDf

def addEmbeddingsToActivities(actsDf, embeddings):
    actsDf["ACTIVITY_SCORES"] = embeddings[-1*(len(actsDf)):]
    return actsDf

def addEmbeddingsToSkills(skillsDf, embeddings):
    skillsDf["SKILL_SCORES"] = embeddings[-1*(len(skillsDf)):]
    return skillsDf


# Normally tokenzie will be done in the translate function
def buildJobsDF(infile, tokenize=False):
    alljobs = JobOffers.readJsonFile(infile)
    tokenized = []
    if not tokenize:
        tokenized = [x["TOKENIZED_JOBS"] for x in alljobs]
    else:
        for i,jo in enumerate(alljobs):
            try:
                cleaned = JobOffers.textCleaner(jo["CONTENT"])
            except:
                print("Failed for job", i)
                cleaned = jo["CONTENT"]
            tokenized.append(JobOffers.sentenceSplitter(cleaned))

    jobsDf = pd.DataFrame(np.array(([x["ID"] for x in alljobs], 
                        [x["ISCO"] for x in alljobs], 
                        tokenized,
                        [x["CONTENT"] for x in alljobs], 
                        [x["TRANSLATED_JOBS" if "TRANSLATED_JOBS" in x else "CONTENT"] for x in alljobs], 
                        [np.array(x["KANTONE"].split(";")) for x in alljobs], 
                        [x["FIRMENGROESSE"] for x in alljobs]
                        ), 
                    dtype='object').T, 
                    columns = ["JOB_ID", "ISCO", "JOB_CONTENT", "ORIGINAL_CONTENT", "TRANSLATED_JOBS", "CANTON", "COMPANY_SIZE"])
    return jobsDf

def buildSkillsDF():
    skillsDf = pd.read_csv(os.path.expanduser(f"{basefolder}/Data/SkillsV26.csv"), sep=";", error_bad_lines=False)
    return skillsDf

def averageNSentences(row, n):
    return np.average(np.sort(row["DISTANCES"])[:n])

def bestMatchReturn(row, results, df, label, match, column_name) :
    bestMatch = results[results["JOB_ID"] == row["JOB_ID"]]
    minEntry = bestMatch[match].idxmin()
    rowMatch = bestMatch.loc[minEntry,:]  
    #return actsDf[actsDf["ACTIVITY_ID"] == rowMatch["ACTIVITY_ID"]]["ACTIVITY_ID"]
    content = df[df[label] == rowMatch[label]]                  
    return content[column_name].iloc[0]  

def bestMatchAttributeReturn(row, results, match, label) :
    bestMatch = results[results["JOB_ID"] == row["JOB_ID"]]
    minEntry = bestMatch[match].idxmin()
    rowMatch = bestMatch.loc[minEntry,:]  
    return rowMatch[label]

def bestNMatchReturn(row, results, df, label, match, column_name, num, cutoff=2.) :
    jobMatch = results[results["JOB_ID"] == row["JOB_ID"]]
    # Find last match < cutoff
    sortedArgs = jobMatch[match].argsort()
    limit = np.count_nonzero(jobMatch[match] < cutoff)
    minEntries = sortedArgs[:min(num, limit)]
    rowMatch = jobMatch.iloc[minEntries,:]
    content = [df[df[label] == x][column_name].iloc[0] for x in rowMatch[label]]
    return content

def minDistanceFinder(row) :
    return min(row["DISTANCES"])

def minDistanceSentence(row, jobs) : 
    minDistanceIdx = np.argmin(row["DISTANCES"])
    df = jobs[jobs["JOB_ID"] == row["JOB_ID"]]
    if df.empty:
        return ""
    content = df.iloc[0]["JOB_CONTENT"]
    return content[minDistanceIdx]

def distCalc(row, label):
    return spatial.distance.cdist(np.expand_dims(row[label], axis=1).T,
                                  np.array(row["JOB_SCORES"]), metric="cosine")[0]

model = hub.load(f"{basefolder}/universal-sentence-encoder_4/")
def embed(input):
  return model(input)

  
