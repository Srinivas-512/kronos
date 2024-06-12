from pymongo import MongoClient
import pandas as pd
import os
from datetime import datetime
import re


def path_parser(path):
    if path=="" or path==" ":
        return []
    pattern1 = re.compile(r'\|id_([^\\]*)\\')
    pattern2 = re.compile(r'\|t_([^\\]*)\\')
    pattern3 = re.compile(r'\|n_([^\\]*)\\')
    id_list = re.findall(pattern1,path)
    t_list = re.findall(pattern2,path)
    n_list = re.findall(pattern3,path)
    res=[[i,j,k]for i,j,k in zip(id_list,t_list,n_list)]
    return res

def combine(paths):
    text=""
    for path in paths:
        if path[0]!="":
            text+=f" id {path[0]},"
        if path[1]!="":
            text+=f" t {path[1]},"
        if path[2]!="":
            text+=f" n {path[2]},"
        text+=" ; "
    return text[:-4]

def unhash(text):
    pattern=f"#\w+#"
    return re.sub(pattern,"",text)

def parse(event_path):
    return unhash(combine(path_parser(event_path)))

def combine_data_attribs(data_attribs):
    attrib = ""
    for attribute in data_attribs:
        attrib += attribute['name']
        attrib += attribute['value']
        attrib += ";;"
    return attrib

# Connect to MongoDB
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client.raw_events_db

# Define the Aggregation Pipeline
pipeline = [
    {
        '$lookup': {
            'from': 'events_client_extended',
            'localField': 'event_id',
            'foreignField': '_id',
            'as': 'event_details'
        }
    },
    {
        '$unwind': '$event_details'
    },
    {
        '$project': {
            'event_time': 1,
            'activity_discovered_name': '$activity_specifications.activity_discovered_name',
            'data_attributes': '$data_attributes',
            'application_id': 1,
            'title': '$event_details.title',
            'url': '$event_details.active_url',
            'event_path': '$event_details.specifications.event_path',
            'participant_id': '$participant_id',
            'day_number': {
                '$dayOfYear': '$event_time'
            }
        }
    },
    {
        '$project': {
            'event_time': 1,
            'activity_discovered_name': 1,
            'data_attributes': 1,
            'application_id': 1,
            'title': 1,
            'url': 1,
            'event_path': 1,
            'case_id': {
                '$concat': [
                    {'$toString': '$participant_id'},
                    '/',
                    {'$toString': '$day_number'}
                ]
            }
        }
    }
]

# Execute the Aggregation
collection = db.event_activities_client_extended
cursor = collection.aggregate(pipeline)

# Convert to DataFrame
df = pd.DataFrame(list(cursor))
df['event_time'] = pd.to_datetime(df['event_time'])

df.drop(columns=['_id'], axis=1, inplace=True)

df['parsed_event_path'] = df['event_path'].apply(parse)

df['data_attributes'] = df['data_attributes'].apply(combine_data_attribs)

df.drop(columns=['data_attributes'], axis=1, inplace=True)

df.to_csv("/home/srinivasan/Skan/Models/kronos_curr/data/cisive_data.csv", index=False)
