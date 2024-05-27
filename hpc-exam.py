import numpy as np
import os
from mpi4py import MPI
import util as ut
import json
from datatypes import ResponseMessage
import threading
import pyopencl as cl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
info = MPI.INFO_ENV
service = 'hpc'
debug = False
path = '/nfs/'

def receive(comm,pn,stats,results,cols,resultsrows):
    data = None
    while data != "stop":
        try:
            data = json.loads(comm.recv(source=pn))
            stats.append(data)
            rowdata = ut.readMatrixFromFile(data["result"])
            row = int(data["row"])
            for j in range(cols):
                results[row*cols+j]=rowdata[j]
            #print("l:"+str(len(stats))+" r:"+str(resultsrows))
        except:
            print(f"errore in {pn}")
            break
            #print("l:"+str(len(stats))+" r:"+str(resultsrows))
            #print(stats)


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
    
    n = int(input("A rows:"))
    m = int(input("A columns/B rows:"))
    p = int(input("B columns:"))
    max = int(input("Maximum:"))

    results = []
    stats = []
    debug = True

    a = ut.initRandomMatrix('A',n,m,max,debug)
    b = ut.initRandomMatrix('B',m,p,max,debug)
    c = ut.initZeroMatrix('C',n,p,debug)


    for i in range(size-1):
        thread = threading.Thread(target=receive, args=(comm,i+1,stats,c,p,n,))
        thread.name="rank-"+str(i+1)+"-receiver"
        threads.append(thread)
        thread.start()
        #print("avviato "+thread.name)
    

    t0 = ut.current_milli_time()
    filenameb = path+str(t0)+'-b.json'
    filenamec = path+str(t0)+'-c.json'
    filenames = path+str(t0)+'-stats.json'
    
    ut.writeMatrixToFile(b,filenameb)
    messages={}
    for i in range(size):
        messages[i]={
            "a":[],
            "b":None
        }


    for i in range(n):
        destproc = 1 + i % (size-1) # the value 0 is used for the master process
        filename = path+str(t0)+'-a-r'+str(i).zfill(5)+'-p'+str(destproc).zfill(2)+'.json'
        r = a.reshape(n, m)[i]
        ut.writeMatrixToFile(r,filename)
        messages[destproc]["a"].append(filename)
        messages[destproc]["b"] = filenameb
    
    t0 = ut.current_milli_time()
    comm.bcast(json.dumps(messages),0)
    #print(json.dumps(messages))
    for thread in threads:
        thread.join()
    out['t']=ut.current_milli_time()-t0
    out['stats']=stats
    out['results']=results
    ut.writeMatrixToFile(c,filenamec)
    ut.writeStatsToFile(out,filenames)

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
        row = pathtok[2][1:]
        filename = pathtok[0]+"-c-r"+str(row)+'-p'+str(rank).zfill(2)+'.json'
        out.setRow(row)
        out.setResult(filename)
        out.setRank(rank)
    

        #the first matrix is always a row vector of size m columns
        #the second matrix have m rows, so the columns are calulate dividing the readed array lenght by the rows
        #the results is a row vector with the same number of columns of b matrix
        
        n = 1
        m = int(len(a))
        p = int(len(b)/m)
        
        c = np.zeros(p, dtype=np.int32)

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

        vector_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        matrix_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        result_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, c.nbytes)

        prg = cl.Program(ctx, """
        __kernel void multiply(__global const int* matrix,
                __global const int* vector,
                __global int* result,
                const int rows,
                const int cols) {
            int row = get_global_id(0);
            //printf("cols:%d rows:%d ",cols,rows);
            //printf("\\n");
            if (row < rows) {
                int sum = 0;
                for (int col = 0; col < cols; col++) { // related to vector column
                    //printf("v[%d]:%d m[%d]:%d m*v:%d\\n",col,vector[col],col * rows + row,matrix[col * rows + row],matrix[col * rows + row] * vector[col]);
                    sum += matrix[col * rows + row] * vector[col];
                    }
                result[row] = sum;
                }
            //printf("\\n");
            }
        """).build()
        kernel = prg.multiply
        kernel.set_args(matrix_buf, vector_buf, result_buf, np.int32(p), np.int32(m))
        t0 = ut.current_milli_time()
        cl.enqueue_nd_range_kernel(queue, kernel, (m,), None)
        cl.enqueue_copy(queue, c, result_buf)
        queue.finish()
        out.setTime(ut.current_milli_time()-t0)
        ut.writeMatrixToFile(c,filename)

        comm.send(out.toJSON(), dest=0)
    comm.send("stop",dest=0)

MPI.Finalize()