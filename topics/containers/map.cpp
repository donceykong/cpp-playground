#include <map>
#include <iostream>
#include <string>

int main() {
    std::map<int, std::string> robotTasks;

    // Insert tasks
    robotTasks[1] = "Get Drunk. Duh.";
    robotTasks[2] = "Object Detection";
    robotTasks[3] = "Mapping";
    robotTasks[4] = "Go to sleep.";

    // Print all tasks
    for (const auto &task : robotTasks) {
        std::cout << "Task ID: " << task.first << ", Task: " << task.second << std::endl;
    }

    // Access and remove a task
    std::cout << "Task 2: " << robotTasks[2] << std::endl;
    robotTasks.erase(2);

    // Check if task 2 still exists
    if (robotTasks.find(2) != robotTasks.end()) {
        std::cout << "Task 2 still exists" << std::endl;
    } else {
        std::cout << "Task 2 has been removed" << std::endl;
    }

    // Print all tasks
    std::cout << "\nPrinting all robot tasks: " << std::endl;
    for (const auto &task : robotTasks) {
        std::cout << "Task ID: " << task.first << ", Task: " << task.second << std::endl;
    }

    return 0;
}
