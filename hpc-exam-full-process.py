import argparse
import sys
import numpy as np
import time
import os
from mpi4py import MPI
import util as ut
import json
from datatypes import DataMessage, ResponseMessage
import threading
import pyopencl as cl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
info = MPI.INFO_ENV
service = 'hpc'
debug = False

def receive(comm,i,count,result):
    res = json.loads(comm.recv(source=i))
    ut.log("received %s from %s",res,str(i))
    result[int(res["row"])]=res
    count = count + 1
    #print(result)



'''
parser = argparse.ArgumentParser(description='hpc exam - MPI Test')
parser.add_argument('-d',help='Enable debug messages')
if rank==0:
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
else:
    args = parser.parse_args()

if args.d is None:
    debug = False
else:
    debug = True
'''
if rank == 0:

    ut.log("Starting main process")
    port = MPI.Open_port(info)
    ut.log("opened port: '%s'", port)    
    MPI.Publish_name(service, info, port)
    ut.log("published service: '%s'", service)

    out = {"t":0,"results":None}
    

    while True:
        threads = []
        global result
        global count
        result = {}
        count = 0

        for i in range(size-1):
            thread = threading.Thread(target=receive, args=(comm,i+1,count,result,))
            thread.start()
            threads.append(thread)
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
        

        t0 = ut.current_milli_time()
        for i in range(n):
            r = a.reshape(n, m)[i]
            data = DataMessage(i,r.tolist(),b.tolist())
            processMessage=data.toJSON()
            destproc = 1 + i % (size-1) # the value 0 is used for the master procecc
            comm.send(processMessage, dest=destproc)
            if debug:
                ut.log("Sent to %s %s",str(destproc),processMessage)
        while count != n:
            pass
        for thread in threads:
            thread.join()
        out['t']=ut.current_milli_time()-t0
        out['results']=result
        print(out)
        action = input("Stop (y/n)?")
        if action=="y":
            for i in range(n):
                destproc = 1 + i % (size-1) 
                comm.send("stop",dest=destproc)
            break

else: # if rank is different then 0 this is a calculator node
    port = None
    ut.log("looking-up service '%s'", service)
    while port is None:
        try:
            port = MPI.Lookup_name(service)
        except:
            pass
    ut.log("service located  at port '%s'", port)
    ut.log('waiting for data ')
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    out = ResponseMessage(result=None,type=None,time=0,row=None)

    while True:
        message = comm.recv(source=0).strip()
        if message == 'stop':
            break
        messageObj = json.loads(message)
        out.setRow(messageObj["row"])

        a = np.array(messageObj["a"])
        b = np.array(messageObj["b"])

        #the first matrix is always a row vector of size 1xn
        #the second matrix have n rows, so the columns are calulate dividing the readed array lenght by the rows
        #the results is a row vector with the same number of columns of b matrix

        n = 1
        m = int(len(a))
        p = int(len(b)/m)

        c = np.zeros(p, dtype=np.float64)

        a = a.astype(np.float64)
        b = b.astype(np.float64)

        platforms = cl.get_platforms()
        dev = platforms[0].get_devices(device_type=cl.device_type.GPU)
        if(len(dev)==0):
            dev = platforms[0].get_devices(device_type=cl.device_type.CPU)
            if debug:
                ut.log("GPU is absent, using CPU")
            out.setType("C")
        else:
            if debug:
                ut.log("using GPU")
            out.setType("G")
        ctx = cl.Context(devices=dev)
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

        prg = cl.Program(ctx, """
            __kernel void multiply(ushort n,
            ushort m, ushort p, __global float *a,
            __global float *b, __global float *c)
            {
            int gid = get_global_id(0);
            c[gid] = 0.0f;
            int rowC = gid/p;
            int colC = gid%p;
            __global float *pA = &a[rowC*m];
            __global float *pB = &b[colC];
            for(int k=0; k<m; k++)
            {
                pB = &b[colC+k*p];
                c[gid] += (*(pA++))*(*pB);
            }
            }
            """).build()
        prg.multiply(queue, c.shape, None, np.uint16(n), np.uint16(m), np.uint16(p), a_buf, b_buf, c_buf)
        a_mul_b = np.empty_like(c)
        t0 = ut.current_milli_time()
        cl.enqueue_copy(queue, a_mul_b, c_buf)
        out.setTime(ut.current_milli_time()-t0)
        out.setResult(a_mul_b.tolist())
        print(out.getResult())
        comm.send(out.toJSON(), dest=0, tag=0)
        '''
        if debug:
            print("t:",str(out.getTime()),"ms")
            print ("matrix A:")
            print (a.reshape(n, m))
            print ("matrix B:")
            print (b.reshape(m, p))
            print ("multiplied A*B:")
            print (a_mul_b.reshape(n, p))
        print(out.toJSON())
        '''

