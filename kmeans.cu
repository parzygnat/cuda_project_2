#include <algorithm>
#include <cstdlib>
#include <stdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>


//TODO declare a Datum struct
struct Datum {
    double x{0};
    double y{0};
    double z{0};
};

using Data = std::vector<Datum>;
//TODO create a square function 

double square(double a) {
    return a*a;
}

//TODO create a function that returns squared distance of two Datums

double squared_distance(Datum a, Datum b) {
    return square(a.x - b.x) + square(a.y - b.y) + square(a.z - b.z);
}

// std::vector<Datum> k_means(const std::vector<Datum>& data,
//     size_t k,
//     size_t number_of_iterations) {
//         //TODO randomize k clusters from available Datums

//         //TODO assign each Datum to the nearest cluster

//         //TODO move clusters - calculate sums

//         //TODO move clusters - divide them by number of Datums in each clusters

//         //TODO return means
//     }

void runCPU(Data data, size_t number_of_Datums, size_t number_of_iterations, double grid_max_value)
{
    printf("Starting sequential bfs.\n\n\n");
    auto start = std::chrono::system_clock::now();
    //kmeansCPU(data, number_of_Datums, number_of_iterations, grid_max_value);
    auto end = std::chrono::system_clock::now();
    float duration = 1000.0*std::chrono::duration<float>(end - start).count();
    printf("\n \n\nElapsed time in milliseconds : %f ms.\n\n", duration);
    
}

int main(int argc, char *argv[])
{
    if(argc < 2)
    { 
      printf("Not enough arguments\n 1st argument -> number of Datums to generate \n 2nd argument -> maximal absolute value on grid \n 3rd argument -> number of iterations of the k-means algorithm\n\n");
     return 0;
    }
    size_t number_of_Datums = atoi(argv[1]);
    double grid_max_value = atof(argv[2]);
    size_t number_of_iterations = atoi(argv[3]);
    Data data(number_of_Datums);
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_real_distribution<double> indices(-grid_max_value, grid_max_value);
    for(auto& Datum : data) {
        Datum.x = indices(random_number_generator);
        Datum.y = indices(random_number_generator);
        Datum.z = indices(random_number_generator);
    }
    // Datum PRINTING
    // for(auto& Datum : data) {
    //     printf("x is %f y is %f and z is %f \n", Datum.x, Datum.y, Datum.z);
    // }
    runCPU(data, number_of_Datums, number_of_iterations, grid_max_value);
    // runGpu();
    return 0;
}