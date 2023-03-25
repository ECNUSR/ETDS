''' Timer '''
import time


class AvgTimer:
    ''' AvgTimer '''
    def __init__(self, window=200):
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.toc = 0
        self.tic = 0
        self.start()

    def start(self):
        ''' start time '''
        self.start_time = self.tic = time.time()

    def record(self):
        ''' record time '''
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count
        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0
        self.tic = time.time()

    def get_current_time(self):
        ''' get_current_time '''
        return self.current_time

    def get_avg_time(self):
        ''' get_avg_time '''
        return self.avg_time
