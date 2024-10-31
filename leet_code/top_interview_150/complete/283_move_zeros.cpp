class Solution {
public:
    // void printVector(std::vector<int> myvector) {
    //     for (unsigned i=0; i<myvector.size(); ++i)
    //         std::cout << myvector[i] << ' ';
    //     std::cout << '\n';
    // }

    void moveZeroes(vector<int>& nums) {
        std::vector<int>::iterator it;
        int elts_removed = 0;
        
        for (int iter = 0; iter<nums.size(); ++iter) {
            if (nums[iter] == 0) {
                nums.erase(nums.begin() + iter);
                nums.push_back(0);
                --iter;
            }
            elts_removed++;
            if (elts_removed > nums.size()) {
                break;
            }
        }
    }
};
