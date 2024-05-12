import json
import codecs
import numpy as np
import time

def readMatrixFromFile(filename):
    obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
    return np.array(json.loads(obj_text))

def writeMatrixToFile(matrix,filename):
    json.dump(matrix.tolist(), codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def current_milli_time():
    return round(time.time() * 1000)

def log(msg, *args):
    print (msg % args)

def initRandomMatrix(name,n,m,debug):
    if debug:
        log("Init %s: %sx%s with random values",name,n,m)
    return np.random.randint(4, size=(n*m))

def initZeroMatrix(name,n,m,debug):
    if debug:
        log("Init %s: %sx%s with zeros",name,n,m)
    return np.zeros((n*m), dtype=np.float64)