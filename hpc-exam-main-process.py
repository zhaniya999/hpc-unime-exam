import argparse
import sys
import pyopencl as cl
import numpy as np
import time
import os
import mpi4py
from mpi4py import MPI
import pandas as pd
import util as ut
import json
from datatypes import DataMessage, ResponseMessage

print (sys.argv)

parser = argparse.ArgumentParser(description='hpc exam - MPI Test')
parser.add_argument('-c',help='number of client process')
parser.add_argument('-n',help='A Matrix rows number')
parser.add_argument('-m',help='A Matrix columns number and B Matrix rows number')
parser.add_argument('-p',help='B Matrix columns number')
parser.add_argument('-d',help='Enable debug messages')
args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
c = int(args.c)
n = int(args.n)
m = int(args.m)
p = int(args.p)

if args.d is None:
    debug = False
else:
    debug = True


ut.log("Starting main process for %s clients process",c)


info = MPI.INFO_NULL
port = MPI.Open_port(info)
ut.log("opened port: '%s'", port)

service = 'hpc'
MPI.Publish_name(service, info, port)
ut.log("published service: '%s'", service)


out = {"e":"","t":0,"platform":"","size":0,"rank":0}

a = ut.initRandomMatrix('A',n,m,debug)
a = a.astype(np.float64)
if debug:
    print(a.reshape(n, m))
b = ut.initRandomMatrix('B',m,p,debug)
b = b.astype(np.float64)
if debug:
    print(b.reshape(m, p))
r = ut.initZeroMatrix('R',n,p,debug)
if debug:
    print(r.reshape(n, p))

ut.log("Waiting to send data")
comm = MPI.COMM_WORLD.Accept(port, info, 0)

for i in range(n):
    r = a.reshape(n, m)[i]
    pn = i % c
    data = DataMessage(i,a,b)
    print(data)
    processMessage=json.dumps(data)
    comm.send(processMessage, dest=0, tag=0)
    print(comm)
    if debug:
        ut.log("Sent %s",processMessage)
    comm = MPI.COMM_WORLD.Accept(port, info, 0)
    

