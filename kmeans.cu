#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <chrono>


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
      printf("Not enough arguments\n 1st argument -> number of examples to generate divisible by 8\n 2nd argument -> maximal absolute value on grid \n 3rd argument -> number of iterations of the k-means algorithm\n\n");
     return 0;
    }
    //default number of clusters = 8;
    size_t number_of_examples = atoi(argv[1]);
    double grid_max_value = atof(argv[2]);
    size_t number_of_iterations = atoi(argv[3]);
    Data data(number_of_examples);
    Boundaries.upper_max = grid_max_value;
    Boundaries.upper_min = grid_max_value*0.5;
    Boundaries.lower_max = -grid_max_value;
    Boundaries.lower_min = -grid_max_value*0.5;
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_real_distribution<double> indices_upper(Boundaries.upper_min, Boundaries.upper_max);
    std::uniform_real_distribution<double> indices_lower(Boundaries.lower_min, Boundaries.lower_max);

    for(int i = 0; i < number_of_examples; ++i) {
        if(i < number_of_examples / 8){
        Datum.x = indices_lower(random_number_generator);
        Datum.y = indices_upper(random_number_generator);
        Datum.z = indices_upper(random_number_generator);
        } else if(i < 2*number_of_examples/8) {
        Datum.x = indices_lower(random_number_generator);
        Datum.y = indices_upper(random_number_generator);
        Datum.z = indices_lower(random_number_generator);
        } else if(i < 3*number_of_examples/8) {
        Datum.x = indices_upper(random_number_generator);
        Datum.y = indices_upper(random_number_generator);
        Datum.z = indices_lower(random_number_generator);
        } else if(i < 4*number_of_examples/8) {
        Datum.x = indices_upper(random_number_generator);
        Datum.y = indices_upper(random_number_generator);
        Datum.z = indices_upper(random_number_generator);
        } else if(i < 5*number_of_examples/8) {
        Datum.x = indices_upper(random_number_generator);
        Datum.y = indices_lower(random_number_generator);
        Datum.z = indices_upper(random_number_generator);
        } else if(i < 6*number_of_examples/8) {
        Datum.x = indices_upper(random_number_generator);
        Datum.y = indices_lower(random_number_generator);
        Datum.z = indices_lower(random_number_generator);
        } else if(i < 7*number_of_examples/8) {
        Datum.x = indices_lower(random_number_generator);
        Datum.y = indices_lower(random_number_generator);
        Datum.z = indices_lower(random_number_generator);
        } else if(i < number_of_examples) {
        Datum.x = indices_lower(random_number_generator);
        Datum.y = indices_lower(random_number_generator);
        Datum.z = indices_upper(random_number_generator);
        }
    }
    // Datum PRINTING
    for(auto& Datum : data) {
        printf("x is %f y is %f and z is %f \n", Datum.x, Datum.y, Datum.z);
    }
    runCPU(data, number_of_examples, number_of_iterations, grid_max_value);
    // runGpu();
    return 0;
}