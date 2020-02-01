import time


class Timer:
    def __init__(self, log=f"Started timer"):
        """

        :param log:
        """
        print(log.strip().capitalize() + "...")
        self.restart_timer()

    def restart_timer(self):
        """

        """
        self.time = time.time()

    def get_elapsed_time(self):
        """

        :return:
        """
        return time.time() - self.time

    def get_human_readable_elapsed_time(self):
        """

        :return:
        """
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
