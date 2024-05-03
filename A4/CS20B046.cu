#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void initializeHealth(int *health, int H, int T) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < T) {
        health[i] = H;
    }
}

__global__ void simulateRound(int *xcoord, int *ycoord, int *health, int *score, int T, int round, bool *gameActive, int *localActiveCount) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= T) return;

    if (health[i] <= 0) return;
    __syncthreads();
    int targetIndex = (i + round) % T;
    if (targetIndex == i) return;

    long long dx = xcoord[targetIndex] - xcoord[i];
    long long dy = ycoord[targetIndex] - ycoord[i];

    long long minDist = LONG_LONG_MAX;
    int hitIndex = -1;

    for (int j = 0; j < T; j++) {
        if (j != i && health[j] > 0) {
            long long distX = xcoord[j] - xcoord[i];
            long long distY = ycoord[j] - ycoord[i];
            int flag = 0;
            if(distX*dx < 0 || distY*dy < 0) flag = 1;
            if (distX*dy==distY*dx && flag == 0) {
                long long distSquared = distX * distX + distY * distY;
                if (distSquared < minDist) {
                    minDist = distSquared;
                    hitIndex = j;
                }
            }
        }
    }

    __syncthreads();
    if (hitIndex != -1) {
        atomicSub(&health[hitIndex], 1);
        atomicAdd(&score[i], 1);
    }

    __syncthreads();
    if(health[i] > 0){
        atomicAdd(localActiveCount, 1);
    }
    __syncthreads();
    // printf("After round %d, health %d for %d tank, score %d, targetting %d, hitting %d, minDist %lld, distsquaerd %lld\n", round, health[i], i, score[i], targetIndex, hitIndex, minDist, distSquared);
    if(*localActiveCount <= 1){
        *gameActive = false;
    }
    __syncthreads();
}


//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int *d_xcoord, *d_ycoord, *d_health, *d_score, *localActiveCount;
    bool *d_gameActive;
    int c = 0;

    cudaMalloc(&d_xcoord, T * sizeof(int));
    cudaMalloc(&d_ycoord, T * sizeof(int));
    cudaMalloc(&d_health, T * sizeof(int));
    cudaMalloc(&d_score, T * sizeof(int));
    cudaMalloc(&d_gameActive, sizeof(bool));
    cudaMalloc(&localActiveCount, sizeof(int));

    cudaMemcpy(d_xcoord, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ycoord, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_score, score, T * sizeof(int), cudaMemcpyHostToDevice);

    initializeHealth<<<1, T>>>(d_health, H, T);

    bool gameActive = true;
    cudaMemcpy(d_gameActive, &gameActive, sizeof(bool), cudaMemcpyHostToDevice);

    int round = 0;
    while (gameActive) {
        cudaMemcpy(localActiveCount, &c, sizeof(int), cudaMemcpyHostToDevice);
        simulateRound<<<1, T>>>(d_xcoord, d_ycoord, d_health, d_score, T, round, d_gameActive, localActiveCount);
        cudaDeviceSynchronize();
        cudaMemcpy(&gameActive, d_gameActive, sizeof(bool), cudaMemcpyDeviceToHost);
        round++;
    }

    cudaMemcpy(score, d_score, T * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_xcoord);
    cudaFree(d_ycoord);
    cudaFree(d_health);
    cudaFree(d_score);
    cudaFree(d_gameActive);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}