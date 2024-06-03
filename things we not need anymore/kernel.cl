__kernel void multiply(__global int *matrix, __global int *vector,
                       __global int *result, const int rows, const int cols) {
  int row = get_global_id(0);
  if (row < rows) {
    int sum = 0;
    for (int col = 0; col < cols; col++) {
      printf("m[%d]=%f v[%d]=%f m*v=%f - ", row * cols + col,
             matrix[row * cols + col], col, vector[col],
             matrix[row * cols + col] * vector[col]);
      sum += matrix[row * cols + col] * vector[col];
    }
    result[row] = sum;
    printf("#");
  }
}