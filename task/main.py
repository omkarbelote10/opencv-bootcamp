import threading

from tracker import run_tracker
from count import run_counter

tracker_thread = threading.Thread(target = run_tracker)

counter_thread = threading.Thread(target = run_counter)


tracker_thread.start()
counter_thread.start()

tracker_thread.join()
counter_thread.join()