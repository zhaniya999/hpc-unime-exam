import json
import codecs
import numpy as np
import time

def readMatrixFromFile(filename):
    obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
    return np.array(json.loads(obj_text)).astype(np.int32)

def writeMatrixToFile(matrix,filename):
    json.dump(matrix.tolist(), codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=0)

def current_milli_time():
    return round(time.time() * 1000)

def log(msg, *args):
    print (msg % args)

def initRandomMatrix(name,n,m,max,debug):
    if debug:
        log("Init %s: %sx%s with random values with a max of %s",name,n,m,max)
        matrix = np.random.randint(max, size=(n*m))
        if n < 10 and m < 10:
            print(matrix.reshape(n,m))
    return matrix

def initZeroMatrix(name,n,m,debug):
    if debug:
        log("Init %s: %sx%s with zeros",name,n,m)
    return np.zeros((n*m), dtype=np.float64)