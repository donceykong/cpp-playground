#include <iostream>
#include "Rectangle.hpp"

int main() {
    Rectangle rect(5.0, 3.0);

    std::cout << "Width: " << rect.getWidth() << std::endl;
    std::cout << "Height: " << rect.getHeight() << std::endl;
    std::cout << "Area: " << rect.area() << std::endl;
    std::cout << "Perimeter: " << rect.perimeter() << std::endl;

    // Modify dimensions
    rect.setWidth(10.0);
    rect.setHeight(6.0);

    std::cout << "\nUpdated dimensions:" << std::endl;
    std::cout << "Width: " << rect.getWidth() << std::endl;
    std::cout << "Height: " << rect.getHeight() << std::endl;
    std::cout << "Area: " << rect.area() << std::endl;
    std::cout << "Perimeter: " << rect.perimeter() << std::endl;

    return 0;
}

