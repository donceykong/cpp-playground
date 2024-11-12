/*
 * Example of using std::list in C++
 * 
 * Why std::list?
 * --------------
 * - `std::list` is a doubly linked list, making it efficient for frequent insertions and deletions.
 * - Unlike `std::vector`, which stores elements in a contiguous block of memory, `std::list` stores each element separately, with pointers to the next and previous elements.
 * - This makes `std::list` ideal for cases where:
 *     - You frequently insert or remove elements in the middle or at both ends of the container.
 *     - You don’t need random access (e.g., accessing the 5th element directly).
 * - Each insertion or deletion in `std::list` has constant time complexity O(1), as it only requires updating a few pointers, while `std::vector` might have to shift elements.
 * - However, `std::list` is less efficient for random access and uses more memory due to pointers.
 */

#include <list>
#include <iostream>

int main() {
    // Create a list of integers
    std::list<int> numbers;

    // Insert elements at the back and the front
    numbers.push_back(10);  // Insert at the end
    numbers.push_front(5);  // Insert at the beginning
    numbers.push_back(15);

    // Display all elements
    std::cout << "Elements in the list:" << std::endl;
    for (const int &num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Insert an element in the middle
    auto it = numbers.begin();
    ++it; // Move iterator to the second position
    numbers.insert(it, 7); // Insert 7 before the second element

    // Display all elements after insertion
    std::cout << "After inserting 7 in the middle:" << std::endl;
    for (const int &num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Erase an element
    it = numbers.begin();
    ++it; // Move to the second element
    numbers.erase(it); // Remove the element at the second position

    // Display all elements after erasure
    std::cout << "After erasing the second element:" << std::endl;
    for (const int &num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
