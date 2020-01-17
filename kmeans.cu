#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
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

__device__ float d_square(float a) {
    return a*a;
}

void runGPU(Points points, Points centroids, size_t number_of_examples, float threshold){
    //TODO initialization and CUDAMallocs
    //TODO AoS to SoA
    //TODO cudaFree
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
    if(number_of_examples%NUMBER_OF_CLUSTERS != 0) {
        printf("The number of examples has to be divisible by 8\n\n");
        return 0;
    }
    Points points(number_of_examples);
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_real_distribution<float> indices_upper(grid_max_value*0.5, grid_max_value);
    std::uniform_real_distribution<float> indices_lower(-grid_max_value, -grid_max_value*0.5);

    for(int i = 0; i < number_of_examples; ++i) {
        if(i < number_of_examples / NUMBER_OF_CLUSTERS){
        points[i].x = indices_lower(random_number_generator);
        points[i].y = indices_upper(random_number_generator);
        points[i].z = indices_upper(random_number_generator);
        } else if(i < 2*number_of_examples/NUMBER_OF_CLUSTERS) {
        points[i].x = indices_lower(random_number_generator);
        points[i].y = indices_upper(random_number_generator);
        points[i].z = indices_lower(random_number_generator);
        } else if(i < 3*number_of_examples/NUMBER_OF_CLUSTERS) {
        points[i].x = indices_upper(random_number_generator);
        points[i].y = indices_upper(random_number_generator);
        points[i].z = indices_lower(random_number_generator);
        } else if(i < 4*number_of_examples/NUMBER_OF_CLUSTERS) {
        points[i].x = indices_upper(random_number_generator);
        points[i].y = indices_upper(random_number_generator);
        points[i].z = indices_upper(random_number_generator);
        } else if(i < 5*number_of_examples/NUMBER_OF_CLUSTERS) {
        points[i].x = indices_upper(random_number_generator);
        points[i].y = indices_lower(random_number_generator);
        points[i].z = indices_upper(random_number_generator);
        } else if(i < 6*number_of_examples/NUMBER_OF_CLUSTERS) {
        points[i].x = indices_upper(random_number_generator);
        points[i].y = indices_lower(random_number_generator);
        points[i].z = indices_lower(random_number_generator);
        } else if(i < 7*number_of_examples/NUMBER_OF_CLUSTERS) {
        points[i].x = indices_lower(random_number_generator);
        points[i].y = indices_lower(random_number_generator);
        points[i].z = indices_lower(random_number_generator);
        } else if(i < number_of_examples) {
        points[i].x = indices_lower(random_number_generator);
        points[i].y = indices_lower(random_number_generator);
        points[i].z = indices_upper(random_number_generator);
        }
    }
    Points centroids(NUMBER_OF_CLUSTERS);
    std::uniform_real_distribution<float> indices(0, number_of_examples - 1);
    for(auto& centroid : centroids) {
        centroid = points[indices(random_number_generator)];
    }
    // Datum PRINTING
    // for(auto& Datum : points) {
    //     printf("x is %f y is %f and z is %f \n", Datum.x, Datum.y, Datum.z);
    // }
    
    runCPU(points, centroids, number_of_examples, threshold);
    //runGPU(points, centroids, number_of_examples, threshold);

    // runGpu();
    return 0;
}