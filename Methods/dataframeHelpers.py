import tokenizers
from easynmt import EasyNMT
from . import JobOffers
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import os
import logging 
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def matchToJob(row, jobBoundaries, allEmbeddings):
    first = jobBoundaries[row.name-1] if row.name != 0 else 0
    last = jobBoundaries[row.name]
    return allEmbeddings[first:last]

# Run this if you want to do the translation, otherwise just use the translation directly
def writeTranslatedJobs():
    alljobs = JobOffers.readJsonFile("Data/output_V1.1.json")
    model = EasyNMT('opus-mt')
    alljobs = JobOffers.addTranslation(alljobs, "output_V1.1.translate.json", maxWrite=100)

def buildActivitiesDF():
    activitiesDf = pd.read_csv(os.path.expanduser("~/Desktop/Thesis/Data/Activities.csv"), error_bad_lines=False)
    activitiesDf = activitiesDf.rename(columns={'O*Net\nMétiers verts' : "ACTIVITY_ID", 'O"Net: Détails des activités pour les métiers verts' : "ACTIVITY_CONTENT"})
    slimmedActsDf = activitiesDf[["ACTIVITY_ID", "ACTIVITY_CONTENT"]][~activitiesDf["ACTIVITY_CONTENT"].isna()]
    slimmedActsDf.loc[slimmedActsDf["ACTIVITY_ID"].isna(), 'ACTIVITY_ID'] = slimmedActsDf[slimmedActsDf["ACTIVITY_ID"].isna()].index
    return slimmedActsDf

def buildJobsDF():
    alljobs = JobOffers.readJsonFile("Data/output_V1.1.translate.json")
    jobsDf = pd.DataFrame(np.array(([x["ID"] for x in alljobs], [x["ISCO"] for x in alljobs], [x["TOKENIZED_JOBS"] for x in alljobs] ), dtype='object').T, columns = ["JOB_ID", "ISCO", "JOB_CONTENT"])
    jobsSentences = np.array([y for x in jobsDf["JOB_CONTENT"] for y in x])
    actsDf = buildActivitiesDF()
    activitiesSentences = actsDf["ACTIVITY_CONTENT"].to_numpy()
    allSentences = np.concatenate((jobsSentences, activitiesSentences))
    allEmbeddings = embed(allSentences)
    jobBoundaries = np.cumsum([len(x) for x in jobsDf["JOB_CONTENT"]])
    jobsDf["JOB_SCORES"] = jobsDf.apply(matchToJob, args=[jobBoundaries, allEmbeddings], axis=1)
    return jobsDf

model = hub.load("/Users/klong/Desktop/Thesis/universal-sentence-encoder_4/")
def embed(input):
  return model(input)
