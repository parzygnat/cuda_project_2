#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <float.h>
#include <chrono>
#define NUMBER_OF_CLUSTERS 8

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

Points kmeansCPU(const Points& points, Points centroids, int number_of_examples, int iterations) {
    Points _centroids(centroids);
    std::vector<int> assignments(number_of_examples);
    for(int i = 0; i < iterations; ++i){
        for(int example = 0; example < number_of_examples - 1; ++example) {
            float currentDistance = std::numeric_limits<float>::max();
            int currentCentroid = 0;
            for(int centroid = 0; centroid < NUMBER_OF_CLUSTERS - 1; ++centroid) {
                if(squared_distance(points[example], centroids[centroid]) < currentDistance){
                    currentDistance = squared_distance(points[example], centroids[centroid]);
                    currentCentroid = centroid;
                }
            }
            assignments[example] = currentCentroid;
        }
        std::vector<int> counter(NUMBER_OF_CLUSTERS, 0);
        Points new_centroids(NUMBER_OF_CLUSTERS);
        for(int assignment = 0; assignment < assignments.size() - 1; ++assignment) {
            new_centroids[assignments[assignment]].x += points[assignment].x;
            new_centroids[assignments[assignment]].y += points[assignment].y;
            new_centroids[assignments[assignment]].z += points[assignment].z;
            counter[assignments[assignment]] = counter[assignments[assignment]] + 1;
        }
        for(int centroid = 0; centroid < NUMBER_OF_CLUSTERS - 1; ++centroid) {
            const auto count = std::max<int>(1, counter[centroid]);
            _centroids[centroid].x = new_centroids[centroid].x/count;
            _centroids[centroid].y = new_centroids[centroid].y/count;
            _centroids[centroid].z = new_centroids[centroid].z/count;
        }
        
    }
    return centroids;
    }

void runCPU(Points points, Points centroids, int number_of_examples, int iterations)
{
    printf("Starting sequential kmeans\n");
    auto start = std::chrono::system_clock::now();
    Points result = kmeansCPU(points, centroids, number_of_examples, iterations);
    auto end = std::chrono::system_clock::now();
    printf("\n");
    for (auto i: result)
        std::cout << i.x << ' ' << i.y << ' ' << i.z << "\n";
    printf("\n");

    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\nElapsed time in milliseconds : %f ms.\n\n", duration);
    
}

__device__ float distance_squared(float x1, float x2, float y1, float y2, float z1, float z2) {
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
}
__global__ void move_centroids(Datum* d_centroids, Datum* new_centroids, int* counters, int number_of_clusters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= number_of_clusters) return;
    Datum _centroid = new_centroids[tid];
    const int count = max(1, counters[tid]);
    d_centroids[tid].x = _centroid.x/count;
    d_centroids[tid].y = _centroid.y/count;
    d_centroids[tid].z = _centroid.z/count;
}

__global__ void distances_calculation(Datum* d_points, Datum* d_centroids, Datum* new_centroids, int* counters, int number_of_examples, int number_of_clusters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    if(tid >= number_of_examples) return;
    extern __shared__ Datum local_centroids[];
    float currentDistance = FLT_MAX;
    int currentCentroid = 0;
    //coalesced read
    float _distance;
    Datum _localDatum = d_points[tid];
    float _x = _localDatum.x;
    float _y = _localDatum.y;
    float _z = _localDatum.z;
    if(local_tid < number_of_clusters) {
        local_centroids[local_tid]= d_centroids[tid];
    }
    for(int i = 0; i < number_of_clusters; ++i) {
        _distance = distance_squared(_x, local_centroids[i].x, _y,local_centroids[i].y , _z, local_centroids[i].z);
        if(_distance < currentDistance) {
            currentCentroid = i;
            currentDistance = _distance;
        }
    }

    //Slow but simple.
    // atomicAdd(&new_centroids[currentCentroid].x, _x);
    // atomicAdd(&new_centroids[currentCentroid].y, _y);
    // atomicAdd(&new_centroids[currentCentroid].z, _z);
    // atomicAdd(&counters[currentCentroid], 1);

}

void runGPU(Points points, Points centroids, int iterations, int number_of_examples, int number_of_clusters){
    //TODO initialization and CUDAMallocs
    Datum* d_points;
    Datum* d_centroids;
    Datum* new_centroids;
    int* counters;
    //we will be accessing memory structures concurrently -> AoS makes more sense than SoA
    cudaMallocManaged(&d_points, points.size()*sizeof(Datum));
    cudaMallocManaged(&d_centroids, centroids.size()*sizeof(Datum));
    cudaMallocManaged(&new_centroids, centroids.size()*sizeof(Datum));
    cudaMallocManaged(&counters, centroids.size()*sizeof(int));
    for(int i = 0; i < number_of_examples; ++i) {
        d_points[i] = points[i];
    }
    for(int i = 0; i < number_of_clusters; ++i) {
        d_centroids[i] = centroids[i];
        new_centroids[i].x = 0;
        new_centroids[i].y = 0;
        new_centroids[i].z = 0;
    }
    
    int num_threads = 1024;
    int num_blocks = (number_of_examples + num_threads - 1) / num_threads;
    int mem = number_of_clusters*sizeof(Datum);
    printf("Starting parallel kmeans\n");
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < iterations; ++i) {
        distances_calculation<<<num_threads, num_blocks, mem>>>(d_points, d_centroids, new_centroids, counters, number_of_examples, number_of_clusters);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        move_centroids<<<1, number_of_clusters>>>(d_centroids, new_centroids, counters, number_of_clusters);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\nElapsed time in milliseconds : %f ms.\n\n", duration);

    for (auto i: centroids)
    std::cout << i.x << ' ' << i.y << ' ' << i.z << "\n";
    printf("\n");

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(new_centroids);
    cudaFree(counters);

}

int main(int argc, char *argv[])
{
    if(argc < 2)
    { 
        printf("Not enough arguments\n 1st argument -> number of examples to generate divisible by 8\n 2nd argument -> maximal absolute value on grid \n 3rd argument -> number of iterations\n\n");
        return 0;
    }
    //default number of clusters = 8;
    int number_of_examples = atoi(argv[1]);
    float grid_max_value = atof(argv[2]);
    int iterations = atoi(argv[3]);
    int number_of_clusters = NUMBER_OF_CLUSTERS;
    if(number_of_examples%number_of_clusters != 0) {
        printf("The number of examples has to be divisible by 8\n\n");
        return 0;
    }
    Points points(number_of_examples);
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_real_distribution<float> indices_upper(grid_max_value*0.5, grid_max_value);
    std::uniform_real_distribution<float> indices_lower(-grid_max_value, -grid_max_value*0.5);

    for(int i = 0; i < number_of_examples; ++i) {
        if(i < number_of_examples / number_of_clusters){
        points[i].x = indices_lower(random_number_generator);
        points[i].y = indices_upper(random_number_generator);
        points[i].z = indices_upper(random_number_generator);
        } else if(i < 2*number_of_examples/number_of_clusters) {
        points[i].x = indices_lower(random_number_generator);
        points[i].y = indices_upper(random_number_generator);
        points[i].z = indices_lower(random_number_generator);
        } else if(i < 3*number_of_examples/number_of_clusters) {
        points[i].x = indices_upper(random_number_generator);
        points[i].y = indices_upper(random_number_generator);
        points[i].z = indices_lower(random_number_generator);
        } else if(i < 4*number_of_examples/number_of_clusters) {
        points[i].x = indices_upper(random_number_generator);
        points[i].y = indices_upper(random_number_generator);
        points[i].z = indices_upper(random_number_generator);
        } else if(i < 5*number_of_examples/number_of_clusters) {
        points[i].x = indices_upper(random_number_generator);
        points[i].y = indices_lower(random_number_generator);
        points[i].z = indices_upper(random_number_generator);
        } else if(i < 6*number_of_examples/number_of_clusters) {
        points[i].x = indices_upper(random_number_generator);
        points[i].y = indices_lower(random_number_generator);
        points[i].z = indices_lower(random_number_generator);
        } else if(i < 7*number_of_examples/number_of_clusters) {
        points[i].x = indices_lower(random_number_generator);
        points[i].y = indices_lower(random_number_generator);
        points[i].z = indices_lower(random_number_generator);
        } else if(i < number_of_examples) {
        points[i].x = indices_lower(random_number_generator);
        points[i].y = indices_lower(random_number_generator);
        points[i].z = indices_upper(random_number_generator);
        }
    }
    Points centroids(number_of_clusters);
    std::uniform_real_distribution<float> indices(0, number_of_examples - 1);
    for(auto& centroid : centroids) {
        centroid = points[indices(random_number_generator)];
    }
    // Datum PRINTING
    // for(auto& Datum : points) {
    //     printf("x is %f y is %f and z is %f \n", Datum.x, Datum.y, Datum.z);
    // }
    
    runCPU(points, centroids, number_of_examples, iterations);
    runGPU(points, centroids, number_of_examples, iterations, number_of_clusters);

    return 0;
}