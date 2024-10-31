/*
 *      Code by Doncey Albin
 *      10/30/2024
 */

#include <iostream>

// Containers:
#include <vector>
#include <map> 
#include <list>

#include <algorithm>

class Solution {
public:
    void printVector(std::vector<int> myvector) {
        for (unsigned i=0; i<myvector.size(); ++i)
            std::cout << myvector[i] << ' ';
        std::cout << '\n';
    }

    int removeElement(std::vector<int>& nums, int val) {
        int k = 0; // nums.size();

        //nums.erase(std::remove(nums.begin(), nums.end(), val), nums.end());
        
        // Iterator for nums
        std::vector<int>::iterator it;
        std::vector<int> nums_clean;

        // Using the iterator to go through the vector
        for (it = nums.begin(); it != nums.end(); ++it) {
            int nums_val = *it; // dereference it to get int value
            if (nums_val != val) {
                nums_clean.push_back(nums_val);
                k++;
            }
        }

        std::cout << "int k is: " << k << std::endl;
        nums = nums_clean;
        printVector(nums);

        return k;
    }
};

int main() {
    Solution solution;
    std::vector<int> nums = {23, 4, 3, 2, 2, 3, 1, 1};
    int val = 3;
    solution.removeElement(nums, val);

    return 0;
}
