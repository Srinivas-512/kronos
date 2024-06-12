import lib.event_gen as event_gen
import lib.trace_builder as trace_builder
import pickle
import os


PARENT_DIR = (os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PARENT_DIR,  'data')

class DataGen():
    def __init__(self, context, skip, persona_id, main_var, version, relaxation_window):
        self.context = context
        self.skip = skip
        self.persona_id = persona_id
        self.main_var = main_var
        self.version = version
        self.relax = relaxation_window
    
    def gen_event_log(self):
        # v1_eventlog = event_gen.load_QAData_Version1(self.persona_id)
        # v2_eventlog = event_gen.load_QAData_Version2(self.persona_id)
        # v3_eventlog = event_gen.load_QAData_Version3(self.persona_id)
        eventlog = event_gen.loadData_genVersion(self.persona_id, self.main_var)
        # return v1_eventlog, v2_eventlog, v3_eventlog
        return eventlog
    
    def gen_traces(self, eventlog):
        # v1_traces = trace_builder.reconstruct_incremental_traces_from_eventlogs(v1_eventlog, self.context, self.skip)
        # v2_traces = trace_builder.reconstruct_incremental_traces_from_eventlogs(v2_eventlog, self.context, self.skip)
        # v3_traces = trace_builder.reconstruct_incremental_traces_from_eventlogs(v3_eventlog, self.context, self.skip)
        traces = trace_builder.reconstruct_incremental_traces_from_eventlogs(eventlog, self.context, self.skip, self.relax)
        # return v1_traces, v2_traces, v3_traces
        return traces

    def save_data(self):
        # v1_eventlog, v2_eventlog, v3_eventlog = self.gen_event_log()
        # v1_traces, v2_traces, v3_traces = self.gen_traces(v1_eventlog, v2_eventlog, v3_eventlog)
        eventlog = self.gen_event_log()
        traces = self.gen_traces(eventlog)
        # event_logs = [v1_eventlog, v2_eventlog, v3_eventlog]
        # traces = [v1_traces, v2_traces, v3_traces]
        directory = os.path.join(DATA_DIR, f'{self.persona_id}')
        if not os.path.exists(directory):
            os.makedirs(directory)

        sub_directory = os.path.join(DATA_DIR, f'{self.persona_id}/v{self.version}')
        if not os.path.exists(sub_directory):
            os.makedirs(sub_directory)
        log_path = os.path.join(sub_directory, f'event_log.pkl')
        trace_path = os.path.join(sub_directory, f'traces_{self.context}_{self.skip}.pkl')
        with open(log_path, 'wb') as file1:
            pickle.dump(eventlog, file1)
        with open(trace_path, 'wb') as file2:
            pickle.dump(traces, file2)


