import os
import csv
import pickle 
from pprint import pprint as pp
from toolz.itertoolz import groupby
from lib.helpers import time_me, make_datetime, delta_seconds

FILE_DIR = (os.path.dirname(__file__))
PARENT_DIR = (os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PARENT_DIR,  'data')
# DATA_OUTPUT_DIR = os.path.join(FILE_DIR,'..','data') # where transformed data will be dumped
# DATA_RAW_DIR = os.path.join(DATA_OUTPUT_DIR,'raw') # where unmodified data is expected

LEN_TIMESTAMP = len('0000-00-00 00:00:00')

def drop_consecutive_repeating_trace_events(trace):
    """ Expects a list of events part of a trace. """
    if len(trace) == 0:
        return []
    if len( set([e['m_activity'] for e in trace]) ) == 1:
        return []

    keep_events = []

    # print("===============")
    for e in trace:
        if len(keep_events) == 0:
            keep_events.append(e)
            # print("Keeping ", e['m_case_id'], e['m_activity'])
        elif keep_events[-1]['m_activity'] == e['m_activity']:
            # print("Dropping", e['m_case_id'], e['m_activity'])
            continue
        else:
            # print("Keeping ", e['m_case_id'], e['m_activity'])
            keep_events.append(e)
    
    return keep_events

def mark_start_end_for_sorted_trace(trace):
    """ Expects a list of events part of a trace. """
    assert len(trace) > 1
    for e in trace:
        e['m_start_end'] = None
    trace[0]['m_start_end'] = 'start'
    trace[-1]['m_start_end'] = 'end'
     
def add_delta_seconds_since_first_event_to_trace_events(trace):
    """ Compute delta seconds since first event """
    # trace_0_timestamp = trace[0]['m_timestamp']
    init_timestamp = make_datetime(trace[0]['m_timestamp'])
    for e in trace:
        e['m_delta_seconds'] = delta_seconds(init_timestamp, make_datetime(e['m_timestamp']))



def loadData_genVersion(persona_id, main_var):

    # data_path = DATA_DIR + f"/{persona_id}.csv"
    data_path = (os.path.join(DATA_DIR, f"{persona_id}.csv"))

    with open(data_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        headers = next(csv_reader)
        headers = [h.lower().replace(' ','_') for h in headers]
        events = [ dict(zip(headers, row_values)) for row_values in csv_reader ]
    
    for idx, event in enumerate(events):

        event["m_case_id"] = event["case_id"]
        event["m_timestamp"] = event["event_time"][:LEN_TIMESTAMP]

        # if main_var == 'activity_discovered_name':
        #     event["m_activity"] = event["activity_discovered_name"]+'/'+event['application_name']
        # else:
        #     event["m_activity"] = event["event_path_parsed"]+'/'+event['application_name']
        # print(event)

        event["m_activity"] = event[main_var]+'/'+event['application_name']

    traces = groupby( lambda x:x['m_case_id'], sorted(events, key=lambda x:x['event_time']) )
    final_events = []
    for _trace in traces.values() :
        trace = drop_consecutive_repeating_trace_events(_trace)
        mark_start_end_for_sorted_trace(trace)
        add_delta_seconds_since_first_event_to_trace_events(trace)
        final_events += trace
    
    for e in final_events:
        assert len(e['m_timestamp']) == LEN_TIMESTAMP
    

    return sorted(final_events, key=lambda x:x['event_time'])


