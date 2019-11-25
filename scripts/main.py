#!python3

import sys
import os
import pandas

import parsing
import utils

'''
pip install numpy
pip install pandas
pip install statsmodels
pip install scikit-learn
pip install patsy
pip install scipy
pip install category_encoders
'''

def getFilepath():
    consoleArgs = sys.argv[1:]
    # if no arguments are given, loads the first file in the folder
    path = "../data"
    if(len(consoleArgs) == 0):
        files = [f for f in os.listdir(path) if os.path.isfile(path + "/" + f)]
        dataFilepath = path + "/" + files[0]
    # else loads the specified file
    else:
        dataFilepath = consoleArgs[0]
    return dataFilepath

def splitDataframe(dataframe, trainingPercentage=0.8):
    lastTrainingRow = int(dataframe.shape[0] * trainingPercentage)
    trainingSet = dataframe.loc[:lastTrainingRow , :]
    testSet = dataframe.loc[lastTrainingRow+1: , :]
    return (trainingSet, testSet)

if __name__ == "__main__":
    totalTimer = utils.Timer()
    totalTimer.restartTimer()

    '''
    P A R S I N G
    '''
    # gets the path to the file checking console's input
    dataFilepath = getFilepath()
    # parses the file to a dataframe
    fullDataframe = parsing.parseFileToDataframe(filepath=dataFilepath, printDataframe=True)

    '''
    F E A T U R E
    E N G I N E E R I N G
    '''
    # deletes some unnecessary index features
    # tmpDataframe = fullDataframe.drop(labels=["ID", "title", "storyline", "year"], axis=1)
    print(f"\nEverything done in {totalTimer.getHumanReadableElapsedTime()}")