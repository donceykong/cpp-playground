#include <unordered_map>
#include <iostream>
#include <string>

int main() {
    // Create an unordered_map that maps integer IDs to string tasks
    std::unordered_map<int, std::string> taskMap;

    // Insert elements into the unordered_map
    taskMap[1] = "Navigation";
    taskMap[2] = "Object Detection";
    taskMap[3] = "Path Planning";

    // Insert using insert() method (another way to add elements)
    taskMap.insert({4, "Mapping"});
    taskMap.insert(std::make_pair(5, "Localization"));

    // Accessing elements
    std::cout << "Task with ID 3: " << taskMap[3] << std::endl;

    // Iterating over all elements (order is arbitrary)
    std::cout << "\nAll tasks in the map:" << std::endl;
    for (const auto& pair : taskMap) {
        std::cout << "Task ID: " << pair.first << ", Task: " << pair.second << std::endl;
    }

    // Checking if a key exists using find()
    if (taskMap.find(2) != taskMap.end()) {
        std::cout << "\nTask with ID 2 exists: " << taskMap[2] << std::endl;
    } else {
        std::cout << "\nTask with ID 2 does not exist." << std::endl;
    }

    // Erasing an element by key
    taskMap.erase(3);  // Remove task with ID 3
    std::cout << "\nAfter erasing task with ID 3:" << std::endl;
    for (const auto& pair : taskMap) {
        std::cout << "Task ID: " << pair.first << ", Task: " << pair.second << std::endl;
    }

    return 0;
}
