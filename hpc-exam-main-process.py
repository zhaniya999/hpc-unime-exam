import argparse
import sys
import numpy as np
import time
import os
from mpi4py import MPI
import util as ut
import json
from datatypes import DataMessage, ResponseMessage


parser = argparse.ArgumentParser(description='hpc exam - MPI Test')
#parser.add_argument('-n',help='A Matrix rows number')
#parser.add_argument('-m',help='A Matrix columns number and B Matrix rows number')
#parser.add_argument('-p',help='B Matrix columns number')
parser.add_argument('-d',help='Enable debug messages')
args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
#c = int(args.c)
#n = int(args.n)
#m = int(args.m)
#p = int(args.p)

if args.d is None:
    debug = False
else:
    debug = True


ut.log("Starting main process")


info = MPI.INFO_NULL
port = MPI.Open_port(info)
ut.log("opened port: '%s'", port)

service = 'hpc'
MPI.Publish_name(service, info, port)
ut.log("published service: '%s'", service)


out = {"t":0,"results":None}

while True:
    n = int(input("n:"))
    m = int(input("m:"))
    p = int(input("p:"))

    a = ut.initRandomMatrix('A',n,m,debug)
    a = a.astype(np.float64)
    if debug:
        print(a.reshape(n, m))
    b = ut.initRandomMatrix('B',m,p,debug)
    b = b.astype(np.float64)
    if debug:
        print(b.reshape(m, p))
    #r = ut.initZeroMatrix('R',n,p,debug)
    #if debug:
    #    print(r.reshape(n, p))
    result = {}
    count = 0
    ut.log("Waiting to send data")
    comm = MPI.COMM_WORLD.Accept(port, info, 0)
    #commr = MPI.COMM_WORLD.Accept(port, info, 1)

    t0 = ut.current_milli_time()
    for i in range(n):
        r = a.reshape(n, m)[i]
        data = DataMessage(i,r.tolist(),b.tolist())
        processMessage=data.toJSON()
        comm.send(processMessage, dest=0, tag=0)
        if debug:
            ut.log("Sent %s",processMessage)
        #comm = MPI.COMM_WORLD.Accept(port, info, 0)
    ut.log("Receiving results")
    while count < n:
        res = json.loads(comm.recv(source=0, tag=0))
        result[int(res["row"])]=res
        count = count + 1
    out['t']=ut.current_milli_time()-t0
    out['results']=result
    print(out)
    action = input("Stop (y/n)?")
    if action=="y":
        comm.send("stop",dest=0,tag=0)
        break

