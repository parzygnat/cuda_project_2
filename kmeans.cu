#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <chrono>
#define NUMBER_OF_CLUSTERS 8

//TODO declare a Datum struct
struct Datum {
    double x{0};
    double y{0};
    double z{0};
};

struct Boundaries {
    double upper_max{0};
    double upper_min{0};
    double lower_max{0};
    double lower_min{0};
};

using Points = std::vector<Datum>;
//TODO create a square function 

double square(double a) {
    return a*a;
}

//TODO create a function that returns squared distance of two examples

double squared_distance(Datum a, Datum b) {
    return square(a.x - b.x) + square(a.y - b.y) + square(a.z - b.z);
}

Points kmeansCPU(const Points& points, Points centroids, size_t number_of_examples, size_t number_of_iterations) {
    std::vector<size_t> assignments(number_of_examples);
    for(int i = 0; i < number_of_iterations; ++i){
        //TODO assign each example to the nearest cluster
        for(int example = 0; example < number_of_examples - 1; ++example) {
            double currentDistance = std::numeric_limits<double>::max();
            size_t currentCentroid = 0;
            for(int centroid = 0; centroid < NUMBER_OF_CLUSTERS - 1; ++centroid) {
                if(squared_distance(points[example], centroids[centroid]) < currentDistance){
                    currentDistance = squared_distance(points[example], centroids[centroid]);
                    currentCentroid = centroid;
                }
            }
            assignments[example] = currentCentroid;
        }
        //TODO move clusters - calculate sums
        std::vector<size_t> counter(NUMBER_OF_CLUSTERS, 0);
        Points new_centroids(NUMBER_OF_CLUSTERS);
        for(int assignment = 0; assignment < assignments.size() - 1; ++assignment) {
            new_centroids[assignments[assignment]].x += points[assignment].x;
            new_centroids[assignments[assignment]].y += points[assignment].y;
            new_centroids[assignments[assignment]].z += points[assignment].z;
            counter[assignments[assignment]] = counter[assignments[assignment]] + 1;
        }
        //TODO move clusters - divide them by number of examples in each clusters
        for(int centroid = 0; centroid < NUMBER_OF_CLUSTERS - 1; ++centroid) {
            const auto count = std::max<size_t>(1, counter[centroid]);
            centroids[centroid].x = new_centroids[centroid].x/count;
            centroids[centroid].y = new_centroids[centroid].y/count;
            centroids[centroid].z = new_centroids[centroid].z/count;
        }
        
    }
    //TODO return means
    return centroids;
    }

void runCPU(Points points, Points centroids, size_t number_of_examples, size_t number_of_iterations)
{
    printf("Starting sequential kmeans\n");
    auto start = std::chrono::system_clock::now();
    Points result = kmeansCPU(points, centroids, number_of_examples, number_of_iterations);
    auto end = std::chrono::system_clock::now();
    printf("\n");
    for (auto i: result)
        std::cout << i.x << ' ' << i.y << ' ' << i.z << "\n";
    printf("\n");

    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\nElapsed time in milliseconds : %f ms.\n\n", duration);
    
}

int main(int argc, char *argv[])
{
    if(argc < 2)
    { 
        printf("Not enough arguments\n 1st argument -> number of examples to generate divisible by 8\n 2nd argument -> maximal absolute value on grid \n 3rd argument -> number of iterations of the k-means algorithm\n\n");
        return 0;
    }
    //default number of clusters = 8;
    size_t number_of_examples = atoi(argv[1]);
    double grid_max_value = atof(argv[2]);
    size_t number_of_iterations = atoi(argv[3]);
    if(number_of_examples%NUMBER_OF_CLUSTERS != 0) {
        printf("The number of examples has to be divisible by 8\n\n");
        return 0;
    }
    Points points(number_of_examples);
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_real_distribution<double> indices_upper(grid_max_value*0.5, grid_max_value);
    std::uniform_real_distribution<double> indices_lower(-grid_max_value, -grid_max_value*0.5);

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
    std::uniform_real_distribution<double> indices(0, number_of_examples - 1);
    for(auto& centroid : centroids) {
        centroid = points[indices(random_number_generator)];
    }
    // Datum PRINTING
    // for(auto& Datum : points) {
    //     printf("x is %f y is %f and z is %f \n", Datum.x, Datum.y, Datum.z);
    // }
    runCPU(points, centroids, number_of_examples, number_of_iterations);
    // runGpu();
    return 0;
}