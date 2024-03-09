/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void Convolution( long int *mat, long int *filter, long int* ans, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ long int shared[];

    int local_row = threadIdx.y + k / 2;
    int local_col = threadIdx.x + k / 2;

    long int *shared_filter = shared;
    long int *shared_mat = shared + k * k;

    if (threadIdx.y < k && threadIdx.x < k) {
        shared_filter[threadIdx.y * k + threadIdx.x] = filter[threadIdx.y * k + threadIdx.x];
    }

    for (int i = -k / 2; i <= k / 2; ++i) {
        for (int j = -k / 2; j <= k / 2; ++j) {
            int r = row + i;
            int c = col + j;
            if (r >= 0 && r < m && c >= 0 && c < n) {
                shared_mat[(i + k / 2) * (k + 31) + (j + k / 2)] = mat[r * n + c];
            } else {
                shared_mat[(i + k / 2) * (k + 31) + (j + k / 2)] = 0;
            }
        }
    }

    __syncthreads();

    if (row < m && col < n) {
        long int sum = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                sum += shared_mat[(local_row + i) * (k + 31) + (local_col + j)] * shared_filter[i * k + j];
            }
        }
        ans[row * n + col] = sum;
    }
}


int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/

    long int* d_h_mat;
    long int* d_h_filter;
    long int* d_h_ans;

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(ceil(n / 32.0), ceil(m / 32.0));

    int shared_mem_size = (k * k + (k + 31)*(k + 31)*sizeof(long int));

    cudaMalloc(&d_h_mat, m*n*sizeof(long int));
    cudaMemcpy(d_h_mat, h_mat, m*n*sizeof(long int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_h_filter, k*k*sizeof(long int));
    cudaMemcpy(d_h_filter, h_filter, k*k*sizeof(long int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_h_ans, m*n*sizeof(long int));
    
    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch

    Convolution<<< blocksPerGrid, threadsPerBlock >>>(d_h_mat, d_h_filter, d_h_ans, m, n, k);
    cudaDeviceSynchronize(); 

    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch

    cudaMemcpy(h_ans, d_h_ans, m*n*sizeof(long int), cudaMemcpyDeviceToHost);

    cudaFree(d_h_mat);
    cudaFree(d_h_filter);
    cudaFree(d_h_ans);
    
    
    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}