# hpc-unime-exam

* pip install pyopencl
* pip install pyopencl[pocl]
* pip install pandas
* https://www.hackerearth.com/practice/notes/raspberry-pi-hacks-part-1-building-mpi-for-python-on-a-raspberry-pi-cluster/
* https://nyu-cds.github.io/python-mpi/setup/

In some environment like LINUX could be necessary to substitute the pip command with pip3

* WINDOWS

    * remove, if present, any OpenMPI, Microsoft MPI
    * run python.exe -m pip uninstall mpi4py
    * clear pip cache with python -m pip cache remove *
    * from https://learn.microsoft.com/it-it/message-passing-interface/microsoft-mpi#ms-mpi-downloads download latest version of Microsoft MPI and install it
    * install mpi4py with python.exe -m pip install mpi4py

Test opencl with:

* WINDOWS

    * mpiexec.exe -n 1 python.exe .\test-mpi-matrix-product.py

* LINUX

    * mpiexec -n 1 python3 ./test-mpi-matrix-product.py

Running the mpi/[client,server].py, using ompi-server:
* LINUX:
    * rm -f /tmp/ompi-server.txt
    * killall ompi-server
    * ompi-server -r /tmp/ompi-server.txt
    * mpiexec -ompi-server file:/tmp/ompi-server.txt -np 1 python mpi/server.py
    * mpiexec -ompi-server file:/tmp/ompi-server.txt -np 1 python mpi/client.py

