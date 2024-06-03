from mpi4py import MPI
import threading

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
service = "test"
info = MPI.INFO_ENV
def receive(comm,i):
    while True:
        msg = comm.recv(source=i)
        print(msg)
        if msg == 'stop':
            break

if rank == 0:
    threads = []
    port = MPI.Open_port(info)
    MPI.Publish_name(service, info, port)
    for i in range(size-1):
        thread = threading.Thread(target=receive, args=(comm,i+1,))
        thread.start()
        threads.append(thread)

    for i in range(size*3):
        data = {'target':i%size,'val':i,'key1' : [7, 2.72, 2+3j],'key2' : ( 'abc', 'xyz')}
        #print(f"Sending message {data} from: {name} whit rank: {rank} of size: {size}")
        comm.send(data, dest=i%size)
    #comm.bcast("stop")

    for i in range(size-1):
        data = 'stop'
        comm.send(data, dest=i+1)

    for thread in threads:
        thread.join()
    
    
else:
    while True:
        #data = None
        #data = comm.bcast(data, root=0)
        data = comm.recv(source=0)
        print(f"Received message by: {name} whith rank: {rank} the message is: {data}")
        comm.send(name+":"+str(rank)+" received data",dest=0)
        if data=='stop':
            comm.send('stop',dest=0)
        #stop = comm.bcast(None, root=0)
        #if stop == "stop":
            #print(f"{name} exit after {stop} message")
            #break