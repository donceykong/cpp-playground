#include <iostream>
using namespace std;

template <typename T>
T multiply(T a, T n) {
    return a * n;
}



int main() {
    int x = 5, y = 10;
    double a = 5.5, b = 10.5;

    cout << "Integer addition: " << multiply(x, y) << endl;
    cout << "Double addition: " << multiply(a, b) << endl;

    return 0;
}
