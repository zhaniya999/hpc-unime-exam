import sys
import numpy as np
import time
import os
from mpi4py import MPI
import util as ut
import json
from datatypes import ResponseMessage
import threading
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.tools import SVMAllocator, SVMPool

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
info = MPI.INFO_ENV
service = 'hpc'
debug = False
path = '/nfs/'

def receive(comm,i,stats,results):
    while size-1 > len(stats):
        data = json.loads(comm.recv(source=i))
        stats.append(data)
        results[data['row']] = data["result"]
        #print("n:"+str(n)+" len(stats):"+str(len(stats)))
        #res = json.loads(comm.recv(source=i))
        #stat = {"rank":i,            "row":res["row"],                    "time":res["time"],"type":res["type"]}
        #stats[int(res["row"])]=stat
        #ut.log("received result row %s from %s",res["row"],str(i))
        #result[int(res["row"])]=res["result"]
        #print("a-i:"+str(i)+" result:"+str(len(result)))
        #print(f"i:{i} count:{count} result:{result} stats:{stats}")


if rank == 0:

    ut.log("Starting main process")
    port = MPI.Open_port(info)
    ut.log("opened port: '%s'", port)    
    MPI.Publish_name(service, info, port)
    ut.log("published service: '%s'", service)

    out = {"t":0,"results":None,"stats":None}
    
    threads = []
    global results
    global stats
    
    n = int(input("n:"))
    m = int(input("m:"))
    p = int(input("p:"))

    results = [""]*n
    stats = []
    debug = True

    for i in range(size-1):
        thread = threading.Thread(target=receive, args=(comm,i+1,stats,results,))
        thread.name="rank-"+str(i+1)+"-receiver"
        threads.append(thread)
        thread.start()

    a = ut.initRandomMatrix('A',n,m,debug)
    a = a.astype(np.float64)
    b = ut.initRandomMatrix('B',m,p,debug)
    b = b.astype(np.float64)
    

    t0 = ut.current_milli_time()
    filenameb = path+str(t0)+'-b.json'
    
    ut.writeMatrixToFile(b,filenameb)
    messages={}
    for i in range(size):
        messages[i]={
            "a":[],
            "b":None
        }


    for i in range(n):
        destproc = 1 + i % (size-1) # the value 0 is used for the master procecc
        filename = path+str(t0)+'-a-'+str(destproc)+'-'+str(i)+'.json'
        r = a.reshape(n, m)[i]
        ut.writeMatrixToFile(r,filename)
        messages[destproc]["a"].append(filename)
        messages[destproc]["b"] = filenameb
    
    t0 = ut.current_milli_time()
    comm.bcast(json.dumps(messages),0)
    for thread in threads:
        thread.join()
    out['t']=ut.current_milli_time()-t0
    #c = ut.mergeResults(n,m,p,results)
    out['stats']=stats
    print(out)

else: # if rank is different then 0 this is a calculator node
    port = None
    #ut.log("looking-up service '%s'", service)
    while port is None:
        try:
            port = MPI.Lookup_name(service)
        except:
            pass
    #ut.log("service located  at port '%s'", port)
    #ut.log('waiting for data ')
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    out = ResponseMessage(result=None,type=None,time=0,row=None)
    message = json.loads(comm.bcast(None,0))
    b = ut.readMatrixFromFile(message[str(rank)]['b'])
    for patha in message[str(rank)]['a']:
        #print(f"rank:{rank} path:{patha}")
        a = ut.readMatrixFromFile(patha)

        pathtok=patha.split("-")
        row = pathtok[len(pathtok)-1].split(".")[0]
        filename = pathtok[0]+"-c-"+str(rank)+"-"+pathtok[len(pathtok)-1]
        out.setRow(row)
        out.setResult(filename)
        out.setRank(rank)
    

        #the first matrix is always a row vector of size 1xn
        #the second matrix have n rows, so the columns are calulate dividing the readed array lenght by the rows
        #the results is a row vector with the same number of columns of b matrix
        
        n = 1
        m = int(len(a))
        p = int(len(b)/m)

        c = np.zeros(p, dtype=np.float64)

        a = a.astype(np.float64)
        b = b.astype(np.float64)

        print(a)
        print(b)
        
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

        matrix_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        vector_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        result_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, c.nbytes)

        prg = cl.Program(ctx, """
            __kernel void multiply(__global float* matrix,
                __global float* vector,
                __global float* result,
                const int rows,
                const int cols) {
                        int row = get_global_id(0);
                        if (row < rows) {
                            float sum = 0.0f;
                            for (int col = 0; col < cols; col++) {
                                printf("m[%d]=%f v[%d]=%f m*v=%f - ",row * cols + col,matrix[row * cols + col],col,vector[col],matrix[row * cols + col] * vector[col]);
                                sum += matrix[row * cols + col] * vector[col];
                            }
                        result[row] = sum;
                        printf("#");
                    }
                }
            """).build()
        kernel = prg.multiply
        kernel.set_args(matrix_buf, vector_buf, result_buf, np.int32(m), np.int32(p))
        t0 = ut.current_milli_time()
        cl.enqueue_nd_range_kernel(queue, kernel, (m,), None)
        cl.enqueue_copy(queue, c, result_buf)
        queue.finish()
        out.setTime(ut.current_milli_time()-t0)
        ut.writeMatrixToFile(c,filename)

        comm.send(out.toJSON(), dest=0)