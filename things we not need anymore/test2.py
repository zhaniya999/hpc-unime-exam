import pyopencl as cl
import numpy as np

# Setup OpenCL context and queue
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Kernel code
kernel_code = """
__kernel void mat_vec_mult(__global const int* matrix,
                           __global const int* vector,
                           __global int* result,
                           const int rows,
                           const int cols) {
    int row = get_global_id(0);
    printf("cols:%d rows:%d ",cols,rows);
    if (row < rows) {
        int sum = 0;
        for (int col = 0; col < cols; col++) {
        printf("col:%d row:%d m[%d]:%d v[%d]:%d mi*vi:%d #",col,row,col * cols + row,matrix[col * cols + row],col,vector[col],matrix[col * cols + row] * vector[col]);
            sum += matrix[col * cols + row] * vector[col];
        }
        result[row] = sum;
    }
}
"""

# Compile the kernel
program = cl.Program(context, kernel_code).build()

# Create example data
rows, cols = 2,3
matrix = np.random.rand(rows, cols)*40
matrix = matrix.astype(np.int32)
print(matrix)
vector = np.random.rand(cols)*40
vector = vector.astype(np.int32)
print(vector)
result = np.zeros(rows, dtype=np.int32)

# Create buffers
matrix_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
vector_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=vector)
result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, result.nbytes)

# Set kernel arguments and execute
kernel = program.mat_vec_mult
kernel.set_args(matrix_buf, vector_buf, result_buf, np.int32(rows), np.int32(cols))

# Execute the kernel
cl.enqueue_nd_range_kernel(queue, kernel, (rows,), None)

# Copy the result from the device to host
cl.enqueue_copy(queue, result, result_buf)

# Wait for all commands to complete
queue.finish()

# Print the results
print("Matrix:\n", matrix)
print("Vector:\n", vector)
print("Result:\n", result)
