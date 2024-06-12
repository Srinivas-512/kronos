from toolz.itertoolz import groupby
import random


def reconstruct_incremental_traces_from_eventlogs(event_logs, context, skip, relaxation_window):
    """ Creates all incremental versions of traces from all event logs.

    Returns an sorted list of dicts: [ { event_sequence : [ <event dict>, ]
                                         next_activity: <activity>  
                                         last_event_timestamp:
                                         activities_set:
                                         case_seq_key:
                                         case_id:

                                       },]

    Assumption: event logs are already sorted by timestamp
    """

    # Event logs should be already ordered
    grpd_eventlogs = groupby(lambda x:x['m_case_id'], event_logs)

    # Order the traces by the timestamp of their respective first event's timestamp 
    grpd_eventlogs = sorted(grpd_eventlogs.values(), key=lambda x:x[0]['m_timestamp'])

    return _create_incremental_process_variations(grpd_eventlogs, context, skip, relaxation_window) 


def _create_incremental_process_variations(grpd_eventlogs, context, skip, relaxation_window):
    """ Expectes a list of event log lists and return a list of 
        lists of all incremental process variations. """
    process_variations = []
    for trace in grpd_eventlogs:
        final = any([event['m_start_end'] == 'end' for event in trace])
        process_variations += _build_incremental_event_sequences_with_metadata(trace, context, skip, relaxation_window)
    return process_variations


def remove_consecutive_duplicates(lst):
    result = []
    for i in range(len(lst)):
        if i == 0 or lst[i]['m_activity'] != lst[i-1]['m_activity']:
            result.append(lst[i])
    return result
 
def reduce_patterns(lst):
    while True:
        new_lst = remove_consecutive_duplicates(lst)
        if new_lst == lst:
            break
        lst = new_lst
    changed = True
    
    while changed:
        changed = False
        n = len(lst) 
        i = 0
        while i < n - 1:
            for j in range(i + 2, n + 1):
                temp1 = lst[i:j]
                temp2 = lst[j:2*j-i]
                flag = True
                if(len(temp2)==0 or len(temp1)==0) or (len(temp1)!=len(temp2)):
                    flag = False
                else:
                    for dict1, dict2 in zip(temp1, temp2):
                        if dict1['m_activity'] != dict2['m_activity']:
                            flag =  False
                            break
                if flag == True:
                    lst = lst[:i] + lst[j:]
                    changed = True
                    n = len(lst)
                    break
            i += 1
    return lst


def _build_incremental_event_sequences_with_metadata(events, context, skip, relaxation_window):
    seqs = []
    size = len(events)
    seen_keys = set()

    for i in range(1, size):
        if i - context + 1 >=0:
            context_events = []
            for j in range(context):
                if j%(skip+1)==0 and i-j>=0:
                    # if j!=0 and events[i-j]['m_activity']==context_events[-1]['m_activity']:
                    #     continue
                    context_events.append(events[i - j])
            context_events = context_events[::-1]
            for j in range(len(context_events)):
                context_events[j]['data_attributes'] = context_events[-1]['data_attributes']
            next_activity = 'END'
            if i!=size-1:
                next_event = events[i + 1]
                next_activity = next_event['m_activity']
            next_activities = []
            j = 1
            while i+j < size-1 and j <=relaxation_window:
                next_activities.append(events[i+j]['m_activity'])
                j+=1
            next_activities = next_activities + ['END'] * (relaxation_window - len(next_activities))

            # context_events = reduce_patterns(context_events)

            if len(context_events)>1:
                case_seq_key = build_case_sequence_key(context_events)

            if len(context_events)>1 and case_seq_key not in seen_keys:
                seqs.append({
                    "event_sequence": context_events,
                    "next_activity": next_activity,
                    "next_activities": next_activities,
                    "last_event_timestamp": context_events[-1]['m_timestamp'],
                    "activities_set": {e['m_activity'] for e in context_events},
                    "activity_sequence_tuple": tuple([e['m_activity'] for e in context_events]),
                    'case_seq_key': build_case_sequence_key(context_events),
                    'case_id': events[0]['m_case_id']
                })
                seen_keys.add(case_seq_key)

        else:
            prev_events = events[:i+1]
            context_events = []
            for j in range(i):
                if j%(skip+1)==0:
                    # if j!=0 and events[i-j]['m_activity']==context_events[-1]['m_activity']:
                    #     continue
                    context_events.append(prev_events[i-j])

            context_events = context_events[::-1]
            next_activity = 'END'
            if i!=size-1:
                next_event = events[i + 1]
                next_activity = next_event['m_activity']
            
            next_activities = []
            j = 1
            while i+j < size-1 and j <=relaxation_window:
                next_activities.append(events[i+j]['m_activity'])
                j+=1
            next_activities = next_activities + ['END'] * (relaxation_window - len(next_activities))

            # context_events = reduce_patterns(context_events)

            if len(context_events)>1:
                case_seq_key = build_case_sequence_key(context_events)

            if len(context_events)>1 and case_seq_key not in seen_keys:
                seqs.append({
                    "event_sequence": context_events,
                    "next_activity": next_activity,
                    "next_activities": next_activities,
                    "last_event_timestamp": context_events[-1]['m_timestamp'],
                    "activities_set": {e['m_activity'] for e in context_events},
                    "activity_sequence_tuple": tuple([e['m_activity'] for e in context_events]),
                    'case_seq_key': build_case_sequence_key(context_events),
                    'case_id': events[0]['m_case_id']
                })
                seen_keys.add(case_seq_key)

    # Ensure that all keys are unique
    unique_keys = set(seq['case_seq_key'] for seq in seqs)
    if len(unique_keys) != len(seqs):
        print(f"Duplicate keys found: {len(seqs) - len(unique_keys)} duplicates")
        raise ValueError("Duplicate case_seq_key found in sequences")
    
    print("num traces  -", len(seqs))

    # seqs = reduce_patterns(seqs)
    
    return seqs

def build_case_sequence_key(events):
    case_id = events[0]['m_case_id']
    acts = [ e['m_activity'] for e in events ]
    return ';;'.join([case_id, '::'.join(acts)])
