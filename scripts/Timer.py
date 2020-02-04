import time


class Timer:
    def __init__(self, log=f"Started timer"):
        """

        :param log:
        """
        print(log.strip()[0].capitalize() + log.strip()[1:] + "...")
        self.restart_timer()

    def restart_timer(self):
        self.time = time.time()

    def get_elapsed_time(self):
        return time.time() - self.time

    def get_human_readable_elapsed_time(self):
        elapsed_time = self.get_elapsed_time()
        if elapsed_time < 120:
            return str('%.2f' % (time.time() - self.time)) + " seconds"
        else:
            return str('%.2f' % ((time.time() - self.time) / 60)) + " minutes"

    def end_timer(self, log="\t...done"):
        """
        :param log:
        """
        log = log.strip()
        print("\t..." + log[0].lower() + log[1:] + f" in {self.get_human_readable_elapsed_time()}")
