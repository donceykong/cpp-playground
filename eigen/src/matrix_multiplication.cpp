#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d mat1;
    Eigen::Matrix2d mat2;

    // Initialize the matrices
    mat1 << 1, 2, 
            3, 4;
    mat2 << 5, 6, 
            7, 8;

    // Perform matrix multiplication
    Eigen::Matrix2d matmul_1_2 = mat1 * mat2;

    // Print the result
    std::cout << "matrix multiplication result: \n" << matmul_1_2 << std::endl;

    return 0;
}
