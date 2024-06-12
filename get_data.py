import csv
from bson import ObjectId
from pymongo import MongoClient
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

# MongoDB connection details
# client = MongoClient('mongodb://localhost:27017/')


# from pymongo import MongoClient 
client = MongoClient('mongodb://127.0.0.1:27017/')
raw_events_db = client['raw_events_db']
# Collections
stats_final_collection = raw_events_db['stats_final']
event_activities_collection = raw_events_db['event_activities']
events_collection = raw_events_db['events']

# personas = ['662b3cc117c9b8105116d4af']
# Specify the persona_id (replace with the actual ObjectId you want to query)

def get_the_data(persona):
    persona_id = ObjectId(persona)  # Change this to the desired persona_id

    # Query to get the document for the specified persona_id
    persona_doc = stats_final_collection.find_one({'persona_id': persona_id})

    if persona_doc:
        cases = persona_doc.get('cases', [])
        data = []

        for case in cases:
            case_id = case.get('case_id')
            case_events = []

            for app in case.get('applications', []):
                application_name = app.get('application_name')
                event_ids = app.get('event_ids', [])

                # Collect and store event details
                for event_id in event_ids:
                    # Get event_time and activity_discovered_name from event_activities_collection
                    event_activity = event_activities_collection.find_one({'event_id': event_id})
                    if event_activity:
                        event_time = event_activity.get('event_time')
                        data_attribs = event_activity.get('data_attributes')
                        attrib = ""
                        for attribute in data_attribs:
                            attrib += attribute['name']
                            attrib += attribute['value']
                            attrib += ";;"
                        activity_discovered_name = event_activity.get('activity_specifications', {}).get('activity_discovered_name')

                        # Get title and active_url from events_collection
                        event = events_collection.find_one({'_id': event_id})
                        if event:
                            title = event.get('title')
                            active_url = event.get('active_url')
                            event_path = event.get('specifications', {}).get('event_path')
                            parsed_event_path = unhash(combine(path_parser(event_path)))

                            # Append the data
                            case_events.append({
                                'case_id': case_id,
                                'event_id': event_id,
                                'event_time': event_time,
                                'application_name': application_name,
                                'title': title,
                                'active_url': active_url,
                                'activity_discovered_name': activity_discovered_name,
                                'data_attributes': attrib,
                                'event_path': event_path,
                                'parsed_event_path' : parsed_event_path
                            })

            # Sort events by event_time within each case
            sorted_case_events = sorted(case_events, key=lambda x: x['event_time'])
            data.extend(sorted_case_events)

        # Write the data to a CSV file
        csv_file = f'/home/srinivasan/Skan/Models/kronos_curr/data/{persona_id}.csv'
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['case_number', 'event_time', 'application_name', 'title', 'active_url', 'activity_discovered_name', 'data_attributes', 'event_path', 'parsed_event_path'])
            for row in data:
                writer.writerow([
                    row['case_id'], row['event_time'], row['application_name'],
                    row['title'], row['active_url'], row['activity_discovered_name'], row['data_attributes'],
                    row['event_path'], row['parsed_event_path']
                ])

        print(f"CSV file '{csv_file}' created successfully.")
    else:
        print(f"No data found for persona_id: {persona_id}")
