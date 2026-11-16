#include <cmath>
#include <iostream>

int main()
{
    long long input;
    std::cout << "Enter an integer: ";
    if (!(std::cin >> input))
    {
        return 1;
    }

    if (input <= 0)
    {
        std::cout << 0 << '\n';
        return 0;
    }

    const unsigned long long number = static_cast<unsigned long long>(input);
    auto root = static_cast<unsigned long long>(std::sqrt(static_cast<long double>(number)));

    while ((root + 1) <= number / (root + 1))
    {
        ++root;
    }
    while (root > number / root)
    {
        --root;
    }

    const unsigned long long lowerSquare = root * root;
    const unsigned long long upperRoot = root + 1;
    const unsigned long long upperSquare = upperRoot * upperRoot;

    const unsigned long long lowerDistance = number - lowerSquare;
    const unsigned long long upperDistance = upperSquare - number;

    const unsigned long long nearsq = lowerDistance <= upperDistance ? lowerSquare : upperSquare;

    const long double near_root = std::sqrt(static_cast<long double>(nearsq));

    const long double perfect_root = std::sqrt(static_cast<long double>(number));

    const long double error = std::fabs(perfect_root - near_root) / perfect_root;

    std::cout << "The nearest square to " << input << " is " << nearsq << std::endl;

    std::cout << "The approximate square root of " << input << " is " << near_root << std::endl;

    std::cout << "The accurate square root of " << input << " is " << perfect_root << std::endl;

    std::cout << "The error is the approximate and perfect root of " << input << " is " << error << std::endl;

    return 0;
}