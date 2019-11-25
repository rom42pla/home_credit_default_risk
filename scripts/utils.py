import time

class Timer:
    def __init__(self):
        self.time = None

    def restartTimer(self):
        self.time = time.time()

    def getElapsedTime(self):
        return time.time() - self.time

    def getHumanReadableElapsedTime(self):
        elapsedTime = self.getElapsedTime()
        if(elapsedTime < 120):
            return str('%.2f' % (time.time() - self.time)) + " seconds"
        else:
            return str('%.2f' % ((time.time() - self.time) / 60)) + " minutes"