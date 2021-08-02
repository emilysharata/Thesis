import tokenizers
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

def buildActivitiesDF():
    activitiesDf = pd.read_csv(os.path.expanduser("~/Desktop/Thesis/Data/Activities.csv"), error_bad_lines=False)
    activitiesDf = activitiesDf.rename(columns={'O*Net\nMétiers verts' : "ACTIVITY_ID", 'O"Net: Détails des activités pour les métiers verts' : "ACTIVITY_CONTENT"})
    slimmedActsDf = activitiesDf[["ACTIVITY_ID", "ACTIVITY_CONTENT"]][~activitiesDf["ACTIVITY_CONTENT"].isna()]
    slimmedActsDf.loc[slimmedActsDf["ACTIVITY_ID"].isna(), 'ACTIVITY_ID'] = slimmedActsDf[slimmedActsDf["ACTIVITY_ID"].isna()].index
    return slimmedActsDf

def buildActivitiesDFV26():
    slimmedActsDf = pd.read_csv(os.path.expanduser("~/Desktop/Thesis/Data/ActivitiesV26.csv"), sep=";", error_bad_lines=False)
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


def buildJobsDF(infile):
    alljobs = JobOffers.readJsonFile(infile)
    jobsDf = pd.DataFrame(np.array(([x["ID"] for x in alljobs], 
                        [x["ISCO"] for x in alljobs], 
                        [x["TOKENIZED_JOBS"] for x in alljobs],
                        [x["CONTENT"] for x in alljobs]), 
                    dtype='object').T, 
                    columns = ["JOB_ID", "ISCO", "JOB_CONTENT", "ORIGINAL_CONTENT"])
    actsDf = buildActivitiesDF()
    return jobsDf

def buildSkillsDF():
    skillsDf = pd.read_csv(os.path.expanduser("~/Desktop/Thesis/Data/SkillsV26.csv"), sep=";", error_bad_lines=False)
    return skillsDf


model = hub.load("/Users/klong/Desktop/Thesis/universal-sentence-encoder_4/")
def embed(input):
  return model(input)
