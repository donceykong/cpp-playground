// erasing from vector
#include <iostream>
#include <vector>

void printVector (std::vector<int> myvector) {
  for (unsigned i=0; i<myvector.size(); ++i)
    std::cout << ' ' << myvector[i];
  std::cout << '\n';
 
}

int main ()
{
  std::vector<int> myvector;

  // set some values (from 1 to 10)
  for (int i=1; i<=10; i++) myvector.push_back(i);
  std::cout << "Initial vector: ";
  printVector(myvector);

  // erase the first element
  myvector.erase (myvector.begin());
  std::cout << "    - After deleting the first elt: ";
  printVector(myvector);

  // erase the first 3 elements:
  myvector.erase (myvector.begin(),myvector.begin()+3);
  std::cout << "    - After deleting the first three elts: ";
  printVector(myvector);

  return 0;
}

