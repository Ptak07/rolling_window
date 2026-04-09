#include <iostream> 
#include "SlidingMedian.hpp"

int main() { 
    std::cout << "Inicjalizacja okna..." << std::endl; 

    SlidingMedian sm(3);

    sm.update(10.0); 
    sm.update(20.0);

    std::cout << "Mediana z [10, 20]: " << sm.get_median() << std::endl; // Powinno być 15

    sm.update(5.0);
    sm.update(5.0);
    sm.update(5.0);
    sm.update(5.0);
    std::cout << "Mediana z [10, 20, 5]: " << sm.get_median() << std::endl;
}