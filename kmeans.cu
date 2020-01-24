#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <float.h>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Datum {
    float x{0};
    float y{0};
    float z{0};
};

using Points = std::vector<Datum>;

float square(float a) {
    return a*a;
}

float squared_distance(Datum a, Datum b) {
    return square(a.x - b.x) + square(a.y - b.y) + square(a.z - b.z);
}

Points kmeansCPU(const Points& points, Points centroids, int number_of_examples, int iterations, int number_of_clusters) {
    std::vector<int> assignments(number_of_examples);
    for(int i = 0; i < iterations; ++i){
        for(int example = 0; example < number_of_examples; ++example) {
            float currentDistance = FLT_MAX;
            int currentCentroid = 0;
            for(int centroid = 0; centroid < number_of_clusters; ++centroid) {
                if(squared_distance(points[example], centroids[centroid]) < currentDistance){
                    currentDistance = squared_distance(points[example], centroids[centroid]);
                    currentCentroid = centroid;
                }
            }
            assignments[example] = currentCentroid;
        }
        // for(auto i:assignments) {
        //     printf(" %d ", i);
        // }
        // printf("\n");
        std::vector<int> counter(number_of_clusters, 0);
        Points new_centroids(number_of_clusters);
        for(int assignment = 0; assignment < assignments.size(); ++assignment) {
            new_centroids[assignments[assignment]].x += points[assignment].x;
            new_centroids[assignments[assignment]].y += points[assignment].y;
            new_centroids[assignments[assignment]].z += points[assignment].z;
            counter[assignments[assignment]] = counter[assignments[assignment]] + 1;
        }
        for(int centroid = 0; centroid < number_of_clusters; ++centroid) {
            const auto count = std::max<int>(1, counter[centroid]);
            centroids[centroid].x = new_centroids[centroid].x/count;
            centroids[centroid].y = new_centroids[centroid].y/count;
            centroids[centroid].z = new_centroids[centroid].z/count;
        }
        
    }
    return centroids;
    }

void runCPU(Points points, Points centroids, int number_of_examples, int iterations, int number_of_clusters)
{
    printf("\nStarting sequential kmeans\n");
    auto start = std::chrono::system_clock::now();
    Points result = kmeansCPU(points, centroids, number_of_examples, iterations, number_of_clusters);
    auto end = std::chrono::system_clock::now();
    printf("\n");
    for (int i = 0; i < number_of_clusters; i++){
        printf("%f  %f  %f", result[i].x, result[i].y, result[i].z);     printf("\n");}


    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\nElapsed time in milliseconds : %f ms.\n\n", duration);
    
}

__device__ float distance_squared(float x1, float x2, float y1, float y2, float z1, float z2) {
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
}
__global__ void move_centroids(float* d_centroids_x, float* d_centroids_y, float* d_centroids_z, float* d_new_centroids_x, float* d_new_centroids_y, float* d_new_centroids_z, float* d_counters, int number_of_clusters, int prev_blocks) 
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int local_tid = threadIdx.x;
    extern __shared__ float this_centroid_x[];
    float* this_centroid_y = (float*)this_centroid_x + blockDim.x; //our current block dim is our previous gridDim
    float* this_centroid_z = (float*)this_centroid_x + blockDim.x + blockDim.x;
    float* this_centroid_counters = (float*)this_centroid_x + blockDim.x + blockDim.x + blockDim.x;
    bool has_element = local_tid < prev_blocks;

    if(has_element) {
        this_centroid_x[local_tid] = d_new_centroids_x[blockIdx*prev_blocks + local_tid];
        this_centroid_y[local_tid] = d_new_centroids_y[blockIdx*prev_blocks + local_tid];
        this_centroid_z[local_tid] = d_new_centroids_z[blockIdx*prev_blocks + local_tid];
        this_centroid_counters[local_tid] = d_counters[blockIdx*prev_blocks + local_tid];
    }
    else {
        this_centroid_x[local_tid] = 0;
        this_centroid_y[local_tid] = 0;
        this_centroid_z[local_tid] = 0;
        this_centroid_counters[local_tid] = 0;
    }
    __syncthreads();

    //TODO reduce on values -> works only when number of blocks is some power of 2 
    for(int d = blockDim.x/2; d > 0; d>>=1) {
        if(local_tid < d) {
            this_centroid_x[local_tid] += this_centroid_x[local_tid + d];
            this_centroid_y[local_tid] += this_centroid_y[local_tid + d];
            this_centroid_z[local_tid] += this_centroid_z[local_tid + d];
            this_centroid_counters[local_tid] += this_centroid_counters[local_tid + d];
        }
        __syncthreads();
    }

    //assignment of new values
    if(local_tid == 0) {
        const float count = this_centroid_counters[local_tid];
        d_centroids_x[blockIdx.x] = this_centroid_x[local_tid]/count;
        d_centroids_y[blockIdx.x] = this_centroid_y[local_tid]/count;
        d_centroids_z[blockIdx.x] = this_centroid_z[local_tid]/count;
    }
    __syncthreads();
    if(has_element) { 
        d_new_centroids_x[tid] = 0;
        d_new_centroids_y[tid] = 0;
        d_new_centroids_z[tid] = 0;
        d_counters[tid] = 0;
    }
}

__global__ void distances_calculation(float* d_points_x, float* d_points_y, float* d_points_z, float* d_centroids_x, float* d_centroids_y, float* d_centroids_z, float* d_new_centroids_x, float* d_new_centroids_y, float* d_new_centroids_z, float* d_counters, int number_of_examples, int number_of_clusters) 
{
    //this version works on atomics with 7x speedup
    extern __shared__ float local_centroids[];
    float* s_array = (float*)local_centroids + 3 * number_of_clusters;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    bool has_element = tid < number_of_examples;
    int currentCentroid = 0;
    float _x; float _y; float _z; float currentDistance;

    if(has_element) {
        currentDistance = FLT_MAX;
        _x = d_points_x[tid]; 
        _y = d_points_y[tid]; 
        _z = d_points_z[tid];
    }

    if(local_tid < number_of_clusters) {
        local_centroids[local_tid] = d_centroids_x[local_tid];
        local_centroids[local_tid + number_of_clusters] = d_centroids_y[local_tid];
        local_centroids[local_tid + number_of_clusters + number_of_clusters] = d_centroids_z[local_tid];
    }
    __syncthreads();
    if(has_element) {
        for(int i = 0; i < number_of_clusters; ++i) {
            const float _distance = distance_squared(_x, local_centroids[i], _y,local_centroids[i + number_of_clusters] , _z, local_centroids[i + 2*number_of_clusters]);
            if(_distance < currentDistance) {
                currentCentroid = i;
                currentDistance = _distance;
            }
        }
    }

    __syncthreads();

    int offset = blockDim.x; 
    int first = local_tid; // x
    int second = local_tid + offset; // y
    int third = local_tid + 2 * offset; //z
    int fourth = local_tid + 3 * offset; //counters

    for(int i = 0; i < number_of_clusters; ++i) 
    {
        s_array[first] = (has_element && (currentCentroid == i)) ? _x : 0;
        s_array[second] = (has_element && (currentCentroid == i)) ? _y : 0;
        s_array[third] = (has_element && (currentCentroid == i)) ? _z : 0;
        s_array[fourth] = (has_element && (currentCentroid == i)) ? 1 : 0;
        __syncthreads();

        for(int d = blockDim.x/2; d > 0; d>>=1) {
            if(local_tid < d) {
                s_array[first] += s_array[first + d];
                s_array[second] += s_array[second + d];
                s_array[third] += s_array[third + d];
                s_array[fourth] += s_array[fourth + d];
            }
            __syncthreads();
        }

        if(local_tid == 0) {
            d_new_centroids_x[i * gridDim.x + blockIdx.x] = s_array[first];
            d_new_centroids_y[i * gridDim.x + blockIdx.x] = s_array[second];
            d_new_centroids_z[i * gridDim.x + blockIdx.x] = s_array[third];
            d_counters[i * gridDim.x + blockIdx.x] = s_array[fourth];
        }
        __syncthreads();
    }

}

void runGPU(Points points, Points centroids, int number_of_examples, int iterations, int number_of_clusters)
{
    //TODO initialization and CUDAMallocs
    float* d_points_x;
    float* d_points_y;
    float* d_points_z;
    float* d_centroids_x;
    float* d_centroids_y;
    float* d_centroids_z;  
    float* d_new_centroids_x;
    float* d_new_centroids_y;
    float* d_new_centroids_z;
    float* d_counters;
    int num_threads = 1024;
    int num_blocks = (number_of_examples + num_threads - 1) / num_threads;
    //we will be accessing memory structures concurrently -> AoS makes more sense than SoA
    cudaMallocManaged(&d_points_x, points.size()*sizeof(float));
    cudaMallocManaged(&d_points_y, points.size()*sizeof(float));
    cudaMallocManaged(&d_points_z, points.size()*sizeof(float));
    cudaMallocManaged(&d_centroids_x, centroids.size()*sizeof(float));
    cudaMallocManaged(&d_centroids_y, centroids.size()*sizeof(float));
    cudaMallocManaged(&d_centroids_z, centroids.size()*sizeof(float));
    cudaMallocManaged(&d_new_centroids_x, num_blocks*centroids.size()*sizeof(float));
    cudaMallocManaged(&d_new_centroids_y, num_blocks*centroids.size()*sizeof(float));
    cudaMallocManaged(&d_new_centroids_z, num_blocks*centroids.size()*sizeof(float));

    cudaMallocManaged(&d_counters, num_blocks*centroids.size()*sizeof(float));
    for(int i = 0; i < number_of_examples; ++i) {
        d_points_x[i] = points[i].x;
        d_points_y[i] = points[i].y;
        d_points_z[i] = points[i].z;
    }
    for(int i = 0; i < number_of_clusters; ++i) {
        d_centroids_x[i] = centroids[i].x;
        d_centroids_y[i] = centroids[i].y;
        d_centroids_z[i] = centroids[i].z;
        d_new_centroids_x[i] = 0;
        d_new_centroids_y[i] = 0;
        d_new_centroids_z[i] = 0;
    }
    

    int mem = 3*number_of_clusters*sizeof(float) + 4*num_threads*sizeof(float);
    int mem2 = 4*num_threads*sizeof(float);
    printf("Starting parallel kmeans\n");
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < iterations; ++i) {
        distances_calculation<<<num_blocks, num_threads, mem>>>(d_points_x, d_points_y, d_points_z, d_centroids_x, d_centroids_y, d_centroids_z, d_new_centroids_x, d_new_centroids_y, d_new_centroids_z, d_counters, number_of_examples, number_of_clusters);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        //for(int i = 0; i < number_of_clusters; ++i) printf("centroid sums: %f %f %f\n", d_new_centroids_x[i], d_new_centroids_y[i], d_new_centroids_z[i]);
        move_centroids<<<number_of_clusters, num_threads, mem2>>>(d_centroids_x, d_centroids_y, d_centroids_z, d_new_centroids_x, d_new_centroids_y, d_new_centroids_z, d_counters, number_of_clusters, num_blocks);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\nElapsed time in milliseconds : %f ms.\n\n", duration);

    for (int i = 0; i < number_of_clusters; i++){
        printf("%f  %f  %f", d_centroids_x[i], d_centroids_y[i], d_centroids_z[i]);     printf("\n");}


    cudaFree(d_points_x);
    cudaFree(d_points_y);
    cudaFree(d_points_z);
    cudaFree(d_centroids_x);
    cudaFree(d_centroids_y);
    cudaFree(d_centroids_z);  
    cudaFree(d_new_centroids_x);
    cudaFree(d_new_centroids_y);
    cudaFree(d_new_centroids_z);
    cudaFree(d_counters);

}

int main(int argc, char *argv[])
{
    if(argc < 5)
    { 
        printf("Not enough arguments\n 1st argument -> number of examples to generate divisible by 8\n 2nd argument -> maximal absolute value on grid \n 3rd argument -> number of iterations \n 4th argument -> number of clusters\n\n");
        return 0;
    }
    //default number of clusters = 8;
    int number_of_examples = atoi(argv[1]);
    float grid_max_value = atof(argv[2]);
    int iterations = atoi(argv[3]);
    int number_of_clusters = atoi(argv[4]);
    if(number_of_examples < number_of_clusters != 0) {
        printf("The number of examples has to be smaller than number of clusters\n\n");
        return 0;
    }
    Points points(number_of_examples);
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());

    {
    //8 domain generation
        // std::uniform_real_distribution<float> indices_upper(grid_max_value*0.5, grid_max_value);
        // std::uniform_real_distribution<float> indices_lower(-grid_max_value, -grid_max_value*0.5);
        // for(int i = 0; i < number_of_examples; ++i) {
        //     if(i < number_of_examples / number_of_clusters){
        //     points[i].x = indices_lower(random_number_generator);
        //     points[i].y = indices_upper(random_number_generator);
        //     points[i].z = indices_upper(random_number_generator);
        //     } else if(i < 2*number_of_examples/number_of_clusters) {
        //     points[i].x = indices_lower(random_number_generator);
        //     points[i].y = indices_upper(random_number_generator);
        //     points[i].z = indices_lower(random_number_generator);
        //     } else if(i < 3*number_of_examples/number_of_clusters) {
        //     points[i].x = indices_upper(random_number_generator);
        //     points[i].y = indices_upper(random_number_generator);
        //     points[i].z = indices_lower(random_number_generator);
        //     } else if(i < 4*number_of_examples/number_of_clusters) {
        //     points[i].x = indices_upper(random_number_generator);
        //     points[i].y = indices_upper(random_number_generator);
        //     points[i].z = indices_upper(random_number_generator);
        //     } else if(i < 5*number_of_examples/number_of_clusters) {
        //     points[i].x = indices_upper(random_number_generator);
        //     points[i].y = indices_lower(random_number_generator);
        //     points[i].z = indices_upper(random_number_generator);
        //     } else if(i < 6*number_of_examples/number_of_clusters) {
        //     points[i].x = indices_upper(random_number_generator);
        //     points[i].y = indices_lower(random_number_generator);
        //     points[i].z = indices_lower(random_number_generator);
        //     } else if(i < 7*number_of_examples/number_of_clusters) {
        //     points[i].x = indices_lower(random_number_generator);
        //     points[i].y = indices_lower(random_number_generator);
        //     points[i].z = indices_lower(random_number_generator);
        //     } else if(i < number_of_examples) {
        //     points[i].x = indices_lower(random_number_generator);
        //     points[i].y = indices_lower(random_number_generator);
        //     points[i].z = indices_upper(random_number_generator);
        //     }
        // }
    }
    std::uniform_real_distribution<float> indices_general(-grid_max_value, grid_max_value);
    for(int i = 0; i < number_of_examples; ++i) {
         points[i].x = indices_general(random_number_generator);
         points[i].y = indices_general(random_number_generator);
         points[i].z = indices_general(random_number_generator);
    }

    Points centroids(number_of_clusters);
    std::uniform_real_distribution<float> indices(0, number_of_examples - 1);
    for(auto& centroid : centroids) {
        centroid = points[indices(random_number_generator)];
    }
    //Datum PRINTING
    // for(auto& Datum : points) {
    //     printf("x is %f y is %f and z is %f \n", Datum.x, Datum.y, Datum.z);
    // }
    
    runGPU(points, centroids, number_of_examples, iterations, number_of_clusters);
    runCPU(points, centroids, number_of_examples, iterations, number_of_clusters);

    return 0;
}