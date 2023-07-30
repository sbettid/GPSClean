from datetime import datetime

def calculate_timedeltas(timestamps):
    #calculate also time deltas to adjust state transition matrix of the Kalman Filter based on the measured delta
    datetimes = []
    if not (type(timestamps[0]) == datetime):
        for dt in timestamps:
            datetimes.append(datetime.fromisoformat(dt.replace("Z", "")))
    else:
        datetimes = timestamps
    
    time_deltas = []
    for i in range(1, len(datetimes)):
        time_deltas.append((datetimes[i] - datetimes[i-1]).total_seconds())

    return time_deltas