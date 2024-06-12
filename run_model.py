import torch
torch.manual_seed(98327)

import os 
import time
import pickle
import csv
from datetime import timedelta, datetime
from pprint import pprint as pp


import lib.event_loader as eventlogs_loader
import lib.helpers as helpers
import lib.attribute_mapper as mapper

from torch.utils.tensorboard import SummaryWriter

import models.lstmnn as lstmnn_model
from lib.dataset_encoders import BinaryEncodedDataset
import lib.trace_builder as trace_builder

from toolz.itertoolz import groupby
from lib.helpers import time_me 

FILE_DIR = (os.path.dirname(__file__))
SIM_BASEPATH = os.path.join(FILE_DIR,'simulation_results')
PARENT_DIR = (os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PARENT_DIR,  'data')
# os.makedirs(SIM_BASEPATH)



### Helpers
def visualise_incremental_traces(incremental_traces):
    for trace in incremental_traces:
        print([e['m_activity'] for e in trace['event_sequence']], '->', trace['next_activity'], trace['last_event_timestamp'])
        print(trace['case_seq_key'], trace['case_id'])
        assert trace['event_sequence'][-1]['m_timestamp'] == trace['last_event_timestamp']


@time_me
def train_binaryencoded_model(traces, attribute_mapper, attribs2use, categ_vars, conti_vars, log_dir, log_params):

    attribute_mapper = attribute_mapper(traces, categ_vars, conti_vars)

    known_activities = attribute_mapper.get_all_activites()
    print("No. of known activities", len(known_activities)) 


    def split_train_valid(traces):
        # Remember we have multiple (incremental) traces for one case.
        # So lets groupy them, then split and unpack them back to traces.
        grp_by_case = list( groupby(lambda x:x['case_id'], traces).values() )
        grp_by_case = sorted(grp_by_case, key=lambda x:x[0]['event_sequence'][0]['m_timestamp'])
        split = int(len(grp_by_case) * 0.8)
        train = [ t for grp_t in grp_by_case[:split] for t in grp_t ]
        valid = [ t for grp_t in grp_by_case[split:] for t in grp_t ]
        
        return train, valid
            
    traces_train, traces_valid = split_train_valid(traces)
    

    train_dataset = BinaryEncodedDataset(traces_train, attribute_mapper, attribs2use)
    valid_dataset = BinaryEncodedDataset(traces_valid, attribute_mapper, attribs2use)


    observed_sequences = [ item['trace']['activity_sequence_tuple'] for item in train_dataset ]
    unique_known_sequences = set(observed_sequences) 

    print("Number of unique observed sequences", len(unique_known_sequences))
    print("Number of observed sequences", len(observed_sequences))

    start_time = time.perf_counter()

    model = lstmnn_model.training_loop(log_dir, train_dataset, valid_dataset, 
            input_size = train_dataset.input_size(),
            output_size = train_dataset.output_size(),
            log_params = log_params,
            dry_run=False)

    training_time = time.perf_counter()-start_time
    print("[Model Training Time]:",training_time)
    return { "model": model, 
             "known_activities": known_activities, 
             "known_sequences": unique_known_sequences, 
             "data_encoder": train_dataset,
             "training_time": training_time,
             "training_samples": len(traces)
           }


class EventStreamHistory():
    def __init__(self, event_logs, traces):

        self.traces, self.td_split_tstamp = self._init_traces(event_logs, traces)
        self.lookup = { t['case_seq_key']:t for t in self.traces }

        self.min_tstamp = event_logs[0]['m_timestamp']
        self.max_tstamp = event_logs[-1]['m_timestamp']

        self.pd_scope_tstamp = str(helpers.make_datetime(self.td_split_tstamp) + timedelta(days=1))

        assert len(self.traces) == len(self.lookup)


    def next_day(self):
        """ Moves past and future window timestamps """
        self.td_split_tstamp = str(helpers.make_datetime(self.td_split_tstamp) + timedelta(days=1))
        self.pd_scope_tstamp = str(helpers.make_datetime(self.td_split_tstamp) + timedelta(days=1))
        
        print("[Next Day Prediction Window ]", self.td_split_tstamp, self.pd_scope_tstamp)

        if str(self.pd_scope_tstamp) <= self.max_tstamp:
            return True
        else:
            return False
    
    def get_training_set(self):
        traces_training_subset = [ t for t in self.traces if t['last_event_timestamp'] <= self.td_split_tstamp ]
        print("Training samples:", len(traces_training_subset))
        return traces_training_subset

    def get_test_set(self, acts_known2model, seqs_known2model):
        # get from till data
        test_set = []
        unknown_acts = []
        unknown_seqs = set()

        for t in self.traces: 
            if t['last_event_timestamp'] > self.td_split_tstamp and t['last_event_timestamp'] <= self.pd_scope_tstamp:
                if t['activities_set'].issubset(acts_known2model):
                    test_set.append( t )
                    # Have we seen the this seq. (with the knowns acts)?
                    if not (t['activity_sequence_tuple'] in seqs_known2model):
                        unknown_seqs.add(t['activity_sequence_tuple'])
                else:
                    unknown_acts += list(t['activities_set'] - set(acts_known2model))
        return (test_set, list(set(unknown_acts)), list(unknown_seqs))

    def update_predictions(self, preds, model_id):
        for p in preds:
            assert not self.lookup[p['case_seq_key']]['predicted_next_activity']
            self.lookup[p['case_seq_key']]['predicted_next_activity'] = p['predicted_value']
            self.lookup[p['case_seq_key']]['prediction_model_id'] = model_id
            #pp(self.lookup[p['case_seq_key']]) 

    def save(self, filepath, model_history):
        with open(filepath, 'wb') as f:
            pickle.dump({ 'traces': self.traces, 'model_history': model_history }, f)
    
    def _init_traces(self, event_logs, traces):
        
        # Splitting timestamp for the past and future data points
        past_set_size = 0.10
        split_timestamp = event_logs[int(len(event_logs)* past_set_size)]['m_timestamp']
        split_timestamp = f"{helpers.make_date(split_timestamp)} 23:59:59" # extend till end of day
        print("Inital Past Data <=", split_timestamp)

        for t in traces:
            t['predicted_next_activity'] = None
            t['prediction_model_id'] = None

        return (traces, split_timestamp)
    
    

def run_simulation00(events_loader, train_func, attribute_mapper, attribs2use, log_dir, file_log, traces, log_params, categ_vars, conti_vars):
    """ Train once and predict until end """

    min_date, max_date = events_loader.min_max_date()
    print("Min. Date", min_date)
    print("Max. Date", max_date)
    all_event_logs = events_loader.pull_data()
    print("Total number of events", len(all_event_logs))
    
    eshistory = EventStreamHistory(all_event_logs, traces)


    model_id = 0
    model = None
    model_history = {}
 
    # Starting model
    training_traces = eshistory.get_training_set()
    model = train_func(training_traces, attribute_mapper, attribs2use, categ_vars, conti_vars, log_dir, log_params)
    model_history[model_id] = {'training_time': model['training_time'], 'training_samples': model['training_samples'] }

    loops = 0
    while True:
        # Predict future states
        test_traces, unknown_acts, unknown_seqs = eshistory.get_test_set(model['known_activities'], model['known_sequences'])

        preds = predict_for_history(model=model['model'], data_encoder = model['data_encoder'], traces=test_traces)
        eshistory.update_predictions(preds, model_id)

        if loops % 100 == 0:
            print("Pickling Traces")
            eshistory.save(file_log, model_history)
        loops += 1

        if not eshistory.next_day():
            print("Final - Pickling Traces")
            eshistory.save(file_log, model_history)
            return



def run_simulation01(events_loader, train_func, attribute_mapper, attribs2use, log_dir, file_log, traces, log_params, categ_vars, conti_vars):
    """ Retrain on encountering new activities """

    min_date, max_date = events_loader.min_max_date()
    print("Min. Date", min_date)
    print("Max. Date", max_date)
    all_event_logs = events_loader.pull_data(from_date=f"{min_date} 00:00:00", till_date=f"{max_date} 23:59:59")
    print("Total number of events", len(all_event_logs))
    
    eshistory = EventStreamHistory(all_event_logs, traces)


    model_id = 0
    model = None
    model_history = {}
 
    # Starting model
    training_traces = eshistory.get_training_set()
    model = train_func(training_traces, attribute_mapper, attribs2use, categ_vars, conti_vars, log_dir, log_params)
    model_history[model_id] = {'training_time': model['training_time'], 'training_samples': model['training_samples'] }

    loops = 0
    while True:
        # Predict future states
        test_traces, unknown_acts, unknown_seqs = eshistory.get_test_set(model['known_activities'], model['known_sequences'])

        preds = predict_for_history(model=model['model'], data_encoder = model['data_encoder'], traces=test_traces)
        eshistory.update_predictions(preds, model_id)

        if loops % 100 == 0:
            print("Pickling Traces")
            eshistory.save(file_log, model_history)
        loops += 1

        if not eshistory.next_day():
            print("Final - Pickling Traces")
            eshistory.save(file_log, model_history)
            return

        # Update Model 
        if unknown_acts: 
            print("[Retrain]")
            training_traces = eshistory.get_training_set()
            model_id += 1
            model = train_func(training_traces, attribute_mapper, attribs2use, categ_vars, conti_vars, log_dir, log_params)
            model_history[model_id] = {'training_time': model['training_time'], 'training_samples': model['training_samples'] }



def run_simulation02(events_loader, train_func, attribute_mapper, attribs2use, log_dir, file_log, traces, log_params, categ_vars, conti_vars):
    """ Retrain when new sequences (this also include new activities) are observed """

    min_date, max_date = events_loader.min_max_date()
    print("Min. Date", min_date)
    print("Max. Date", max_date)
    all_event_logs = events_loader.pull_data(from_date=f"{min_date} 00:00:00", till_date=f"{max_date} 23:59:59")
    print("Total number of events", len(all_event_logs))
    
    eshistory = EventStreamHistory(all_event_logs, traces)

    model_id = 0
    model = None
    model_history = {}
 
    # Starting model
    training_traces = eshistory.get_training_set()
    
    model = train_func(training_traces, attribute_mapper, attribs2use, categ_vars, conti_vars, log_dir, log_params)
    model_history[model_id] = {'training_time': model['training_time'], 'training_samples': model['training_samples'] }
   
    loops = 0
    while True:
        print(f"Prediction Day {loops}")
        test_traces, unknown_acts, unknown_seqs = eshistory.get_test_set(model['known_activities'], model['known_sequences'])

        preds = predict_for_history(model=model['model'], data_encoder = model['data_encoder'], traces=test_traces)
        eshistory.update_predictions(preds, model_id)

        if loops % 10 == 0:
            print("Pickling Traces")
            eshistory.save(file_log, model_history)
        loops += 1

        if not eshistory.next_day():
            print("Final - Pickling Traces")
            eshistory.save(file_log, model_history)
            return

        if unknown_seqs:
            print("[Retrain]")
            training_traces = eshistory.get_training_set()
            model_id += 1
            model = train_func(training_traces, attribute_mapper, attribs2use, categ_vars, conti_vars, log_dir, log_params)
            model_history[model_id] = {'training_time': model['training_time'], 'training_samples': model['training_samples'] }



def run_simulation03(events_loader, train_func, attribute_mapper, attribs2use, log_dir, file_log, traces, log_params, categ_vars, conti_vars):
    """ Retrain every day """

    min_date, max_date = events_loader.min_max_date()
    print("Min. Date", min_date)
    print("Max. Date", max_date)
    all_event_logs = events_loader.pull_data(from_date=f"{min_date} 00:00:00", till_date=f"{max_date} 23:59:59")
    print("Total number of events", len(all_event_logs))
    
    eshistory = EventStreamHistory(all_event_logs, traces)

    model_id = 0
    model = None
    model_history = {}
 
    # Starting model
    training_traces = eshistory.get_training_set()
    model = train_func(training_traces, attribute_mapper, attribs2use, categ_vars, conti_vars, log_dir, log_params)
    model_history[model_id] = {'training_time': model['training_time'], 'training_samples': model['training_samples'] }
   
    loops = 0
    while True:
        print(f"Prediction Day {loops}")
        test_traces, unknown_acts, unknown_seqs = eshistory.get_test_set(model['known_activities'], model['known_sequences'])

        preds = predict_for_history(model=model['model'], data_encoder = model['data_encoder'], traces=test_traces)
        eshistory.update_predictions(preds, model_id)

        if loops % 10 == 0:
            print("Pickling Traces")
            eshistory.save(file_log, model_history)
        loops += 1

        if not eshistory.next_day():
            print("Final - Pickling Traces")
            eshistory.save(file_log, model_history)
            return

        print("[Retrain]")
        training_traces = eshistory.get_training_set()
        model_id += 1
        model = train_func(training_traces, attribute_mapper, attribs2use, categ_vars, conti_vars, log_dir, log_params)
        model_history[model_id] = {'training_time': model['training_time'], 'training_samples': model['training_samples'] }



def predict_for_history(**kwargs):
    import torch
    from torch.nn.utils.rnn import pack_sequence

    model = kwargs['model']
    model.eval()

    prediction_results = []
    for input in  kwargs['traces']:
        inp_enc = (torch.DoubleTensor( kwargs['data_encoder'].events_to_input(input['event_sequence']) ))
        pred_index = ( model(pack_sequence([inp_enc]))[1].softmax(dim=1).argmax(dim=1))
        class_value = kwargs['data_encoder'].index_to_activity_name(pred_index.item())

        prediction_results.append({ 'case_seq_key': input['case_seq_key'], 
                                    'predicted_value': class_value })
    return prediction_results


def run_simulations(persona_id, context, skip, data_v, attribs, categ_vars, conti_vars):
    attribs2use = attribs

    eventlog_path = os.path.join(DATA_DIR, f'{persona_id}/v{data_v}/event_log.pkl')
    traces_path = os.path.join(DATA_DIR, f'{persona_id}/v{data_v}/traces_{context}_{skip}.pkl')

    with open(eventlog_path, 'rb') as file1:
        event_logs = pickle.load(file1)

    with open(traces_path, 'rb') as file2:
        traces = pickle.load(file2)
    
    loader = eventlogs_loader.genVersionLogs(event_logs)

    results_path = os.path.join(DATA_DIR, f'{persona_id}/results.csv')

    headers = ["context","skip","data_version","update_version","train_loss","valid_loss","accuracy","mcc","f1-macro","f1-weighted","baccu"]

    with open(results_path, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(headers)

        writer.writerow([])
        writer.writerow([])

    log_params0 = [context, skip, data_v, 1, results_path]
    log_params1 = [context, skip, data_v, 2, results_path]
    log_params2 = [context, skip, data_v, 3, results_path]
    log_params3 = [context, skip, data_v, 4, results_path]

    run_simulation00(events_loader = loader,
                     train_func = train_binaryencoded_model, 
                     attribute_mapper = mapper.generalMapper,
                     attribs2use = attribs2use,
                     log_dir = None,
                     file_log = os.path.join(SIM_BASEPATH, 'QA_Data_Version3_simulation_00.pickle'),
                     traces = traces, 
                     log_params = log_params0,
                     categ_vars = categ_vars, 
                     conti_vars = conti_vars
                    )


    run_simulation01(events_loader = loader,
                     train_func = train_binaryencoded_model, 
                     attribute_mapper = mapper.generalMapper,
                     attribs2use = attribs2use,
                     log_dir = None,
                     file_log = os.path.join(SIM_BASEPATH, 'QA_Data_Version3_simulation_01.pickle'),
                     traces = traces,
                     log_params = log_params1,
                     categ_vars = categ_vars, 
                     conti_vars = conti_vars
                    )

    run_simulation02(events_loader = loader,
                     train_func = train_binaryencoded_model, 
                     attribute_mapper = mapper.generalMapper,
                     attribs2use = attribs2use,
                     log_dir = None,
                     file_log = os.path.join(SIM_BASEPATH, 'QA_Data_Version3_simulation_02.pickle'),
                     traces = traces,
                     log_params = log_params2,
                     categ_vars = categ_vars, 
                     conti_vars = conti_vars
                    )

    run_simulation03(events_loader = loader,
                     train_func = train_binaryencoded_model, 
                     attribute_mapper = mapper.generalMapper,
                     attribs2use = attribs2use,
                     log_dir = None,
                     file_log = os.path.join(SIM_BASEPATH, 'QA_Data_Version3_simulation_03.pickle'),
                     traces = traces,
                     log_params = log_params3,
                     categ_vars = categ_vars, 
                     conti_vars = conti_vars
                    )
