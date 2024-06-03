#! /usr/bin/env python

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
while True:
    print("rank:"+str(rank))


def log(msg, *args):
    if rank == 0:
        print (msg % args)

info = MPI.INFO_NULL
service = "pyeval"

if rank==0:
    log('Start')

    port = MPI.Open_port(info)
    log("opened port: '%s'", port)

    MPI.Publish_name(service, info, port)
    log("published service: '%s'", service)

    #MPI.COMM_WORLD.Spawn("./server.py", maxprocs=1)

    root = 0
    log('waiting for client connection...')
    comm = MPI.COMM_WORLD.Accept(port, info, root)
    #comm = MPI.COMM_WORLD.Connect(port, info, root)
    log('client connected...')

    while True:
        done = False
        if rank == root:
            message = comm.recv(source=0, tag=0)
            if message is None:
                done = True
            else:
                try:
                    print ('eval(%r) -> %r' % (message, eval(message)))
                    comm.send(eval(message), dest=0, tag=0)
                except Exception:
                    print ("invalid expression: %s" % message)
        done = MPI.COMM_WORLD.bcast(done, root)
        if done:
            break

    log('disconnecting client...')
    comm.Disconnect()

    log('upublishing service...')
    MPI.Unpublish_name(service, info, port)

    log('closing port...')
    MPI.Close_port(port)
else:
    log("looking-up service '%s'", service)
    port = MPI.Lookup_name(service)
    log("service located  at port '%s'", port)

    root = 0
    log('waiting for server connection...')
    comm = MPI.COMM_WORLD.Connect(port, info, root)
    log('server connected...')

    while True:
        done = False
        if rank == root:
            try:
                message = input('pyeval (quit to exit)>>> ')
                if message == 'quit':
                    message = None
                    
                    done = True
            except EOFError:
                message = None
                done = True
            comm.send(message, dest=0, tag=0)
            message = comm.recv(source=0, tag=0)
            log('received:%s' %message)
        else:
            message = None
        done = MPI.COMM_WORLD.bcast(done, root)
        if done:
            break

    log('disconnecting server...')
    comm.Disconnect()
