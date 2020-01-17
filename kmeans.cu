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

Points kmeansCPU(const Points& points, Points centroids, size_t number_of_examples, float threshold) {
    std::vector<size_t> assignments(number_of_examples);
    float changed = number_of_examples;
    while(changed/number_of_examples > threshold){
        //printf("changed is %f\n", changed);
        changed = 0;
        for(int example = 0; example < number_of_examples - 1; ++example) {
            float currentDistance = std::numeric_limits<float>::max();
            size_t currentCentroid = 0;
            for(int centroid = 0; centroid < NUMBER_OF_CLUSTERS - 1; ++centroid) {
                if(squared_distance(points[example], centroids[centroid]) < currentDistance){
                    currentDistance = squared_distance(points[example], centroids[centroid]);
                    currentCentroid = centroid;
                }
            }
            if(assignments[example] != currentCentroid) ++changed;
            assignments[example] = currentCentroid;
        }
        std::vector<size_t> counter(NUMBER_OF_CLUSTERS, 0);
        Points new_centroids(NUMBER_OF_CLUSTERS);
        for(int assignment = 0; assignment < assignments.size() - 1; ++assignment) {
            new_centroids[assignments[assignment]].x += points[assignment].x;
            new_centroids[assignments[assignment]].y += points[assignment].y;
            new_centroids[assignments[assignment]].z += points[assignment].z;
            counter[assignments[assignment]] = counter[assignments[assignment]] + 1;
        }
        for(int centroid = 0; centroid < NUMBER_OF_CLUSTERS - 1; ++centroid) {
            const auto count = std::max<size_t>(1, counter[centroid]);
            centroids[centroid].x = new_centroids[centroid].x/count;
            centroids[centroid].y = new_centroids[centroid].y/count;
            centroids[centroid].z = new_centroids[centroid].z/count;
        }
        
    }
    return centroids;
    }

void runCPU(Points points, Points centroids, size_t number_of_examples, float threshold)
{
    printf("Starting sequential kmeans\n");
    auto start = std::chrono::system_clock::now();
    Points result = kmeansCPU(points, centroids, number_of_examples, threshold);
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

__global__ void distances_calculation(Datum* d_points, Datum* d_centroids, Datum* new_centroids, size_t* counters, size_t* assignments, size_t number_of_examples, size_t number_of_clusters, size_t* if_changed) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= number_of_examples) return;
    size_t local_tid = blockIdx.x;
    extern __shared__ Datum local_centroids[];
    //coalesced read
    float _distance;
    float _x = d_points[tid].x;
    float _y = d_points[tid].y;
    float _z = d_points[tid].z;

    float currentDistance = FLT_MAX;
    size_t currentCentroid = 0;
    if(local_tid < number_of_clusters) {
        local_centroids[tid]= d_centroids[tid];
    }
    for(int i = 0; i < number_of_clusters; ++i) {
        _distance = distance_squared(_x, local_centroids[i].x, _y,local_centroids[i].y , _z, local_centroids[i].z);
        if(_distance < currentDistance) {
            currentCentroid = i;
            currentDistance = _distance;
        }
    }

    if_changed[tid] = 0;
    if(assignments[tid] != currentCentroid) {
        if_changed[tid] = 1;
        assignments[tid] = currentCentroid;
    }

    printf("im tid %d\n", tid);

      // Slow but simple.
    atomicAdd(&new_centroids[currentCentroid].x, _x);
    atomicAdd(&new_centroids[currentCentroid].y, _y);
    atomicAdd(&new_centroids[currentCentroid].z, _z);
    atomicAdd(&counters[currentCentroid], (size_t)1);

}

void runGPU(Points points, Points centroids, size_t number_of_examples, float threshold, size_t number_of_clusters){
    //TODO initialization and CUDAMallocs
    float changed = number_of_examples;
    Datum* d_points;
    size_t* if_changed;
    Datum* d_centroids;
    Datum* new_centroids;
    size_t* counters;
    size_t* assignments;
    //we will be accessing memory structures concurrently -> AoS makes more sense than SoA
    cudaMallocManaged(&if_changed, points.size()*sizeof(size_t));
    cudaMallocManaged(&d_points, points.size()*sizeof(Datum));
    cudaMallocManaged(&d_centroids, centroids.size()*sizeof(Datum));
    cudaMallocManaged(&new_centroids, centroids.size()*sizeof(Datum));
    cudaMallocManaged(&counters, centroids.size()*sizeof(size_t));
    cudaMallocManaged(&assignments, points.size()*sizeof(size_t));
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
    //while(changed/number_of_examples > threshold) {
        changed = 0;
        distances_calculation<<<num_threads, num_blocks, mem>>>(d_points, d_centroids, new_centroids, counters, assignments, number_of_examples, number_of_clusters, if_changed);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        // move_centroids<<<1, number_of_clusters>>>>();
        // gpuErrchk( cudaPeekAtLastError() );
        // gpuErrchk( cudaDeviceSynchronize() );

    //}

    //TODO cudaFree
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(new_centroids);
    cudaFree(counters);

}

int main(int argc, char *argv[])
{
    if(argc < 2)
    { 
        printf("Not enough arguments\n 1st argument -> number of examples to generate divisible by 8\n 2nd argument -> maximal absolute value on grid \n 3rd argument -> 0-1 threshold for stopping iterating\n\n");
        return 0;
    }
    //default number of clusters = 8;
    size_t number_of_examples = atoi(argv[1]);
    float grid_max_value = atof(argv[2]);
    float threshold = atof(argv[3]);
    size_t number_of_clusters = NUMBER_OF_CLUSTERS;
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
    
    runCPU(points, centroids, number_of_examples, threshold);
    runGPU(points, centroids, number_of_examples, threshold, number_of_clusters);

    return 0;
}