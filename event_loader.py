class EventLogLoader():
    def __init__(self, event_logs):
        self.event_logs = event_logs

    def pull_data(self, from_date=None, till_date=None):
        event_logs = self.event_logs
        assert event_logs != None
        if not from_date and not till_date:
            return event_logs
        elif from_date and till_date:
            return [ e for e in event_logs if e['m_timestamp'] >= from_date and e['m_timestamp'] <= till_date ]
        elif from_date and not till_date:
            return [ e for e in event_logs if e['m_timestamp'] >= from_date ]
        elif not from_date and till_date:
            return [ e for e in event_logs if e['m_timestamp'] <= till_date ]

    def min_max_date(self):
        """ Return min and maximum date from a sorted input of event logs """
        event_logs = self.event_logs
        return (event_logs[0]['m_timestamp'], event_logs[-1]['m_timestamp'])


# Public API
def QAVersion1Logs(event_logs):
    return EventLogLoader(event_logs)

def QAVersion2Logs(event_logs):
    return EventLogLoader(event_logs)

def QAVersion3Logs(event_logs):
    return EventLogLoader(event_logs)

def genVersionLogs(event_logs):
    return EventLogLoader(event_logs)