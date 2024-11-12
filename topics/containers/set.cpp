#include <set>
#include <iostream>

int main() {
    // Create a set of integers
    std::set<int> numbers;

    // Insert elements into the set
    numbers.insert(10);
    numbers.insert(5);
    numbers.insert(15);
    numbers.insert(10); // Duplicate value, won't be added

    // Display all elements (automatically sorted in ascending order)
    std::cout << "Elements in the set:" << std::endl;
    for (const int &num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Check if an element exists
    int toFind = 5;
    if (numbers.find(toFind) != numbers.end()) {
        std::cout << "Element " << toFind << " exists in the set." << std::endl;
    } else {
        std::cout << "Element " << toFind << " does not exist in the set." << std::endl;
    }

    // Erase an element
    numbers.erase(10); // Removes the element with value 10
    std::cout << "After erasing 10:" << std::endl;
    for (const int &num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
