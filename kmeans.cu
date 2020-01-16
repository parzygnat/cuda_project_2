#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

//TODO declare a point struct
struct Point {
    double x{0};
    double y{0};
    double z{0};
};
//TODO create a square function 

double square(double a) {
    return a*a;
}

//TODO create a function that returns squared distance of two points

double squared_distance(Point a, Point b) {
    return square(a.x - b.x) + square(a.y - b.y) + square(a.z - b.z);
}

 using DataFrame = std::vector<Point>;
// DataFrame k_means(const DataFrame& data,
//     size_t k,
//     size_t number_of_iterations) {
//         //TODO randomize k clusters from available points

//         //TODO assign each point to the nearest cluster

//         //TODO move clusters - calculate sums

//         //TODO move clusters - divide them by number of points in each clusters

//         //TODO return means
//     }
using DataFrame = std::vector<Point>;
int main(int argc, char *argv[])
{
    if(argc < 2)
    { 
      printf("Not enough arguments\n");
     return 0;
    }
    size_t number_of_points = atoi(argv[1]);
    double grid_max_value = atoi(argv[2]);
    size_t number_of_iterations = atoi(argv[3]);
    DataFrame data(size_t);
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_real_distribution<double> indices(-grid_max_value, grid_max_value);
    for(auto& point : data) {
        point.x = indices(random_number_generator);
        point.y = indices(random_number_generator);
        point.z = indices(random_number_generator);
    }
    for(auto& point : data) {
        printf("x is %d y is %d and z is %d \n", point.x, point.y, point.z);
    }
    // runCpu();
    // runGpu();
    return 0;
}