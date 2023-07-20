import time

class TimeStats(object):
    def __init__(self) -> None:
        self.reset()

    def __str__(self):
        return str(self._time_dict)
    
    def reset(self):
        self.flag_timestem = {}
        self._time_dict = {
            "set_parameters": 0,
            "fit": 0,
            "evaluate": 0,
            "get_parameters": 0,
        }
    
    def mark_start(self, name):
        self.flag_timestem[name] = time.time()
    
    def mark_end(self, name):
        self._time_dict[name] += time.time() - self.flag_timestem[name]
        del self.flag_timestem[name]
