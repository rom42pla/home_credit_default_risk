#!python3
# -*- coding: utf-8 -*-

import pandas
import os
import ast

import utils

def parseFileToDataframe(filepath, printDataframe = False, clean = True):
    sectionTimer = utils.Timer()
    print(f"Parsing file {filepath} ({'{0:.1f}'.format(os.path.getsize(filepath) * 2**(-20))}Mb)...")
    sectionTimer.restartTimer()

    dataframe = pandas.read_csv(filepath)

    print(dataframe)

    print(f"\t...parsed a dataframe with shape ({dataframe.shape[0]}x{dataframe.shape[1]}) in {sectionTimer.getHumanReadableElapsedTime()}")
    return dataframe