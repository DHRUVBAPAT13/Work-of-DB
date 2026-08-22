#include <iostream>
#include <cmath>

using namespace std;

double addition(double a, double b);
double subtraction(double a, double b);
double multiplication(double a, double b);
double division(double a, double b);
double exponent(double a, double b);
double factorial(double a);
double square_root(double a);
double square(double a);
double cube_root(double a);
double cube(double a);
bool isPrime(double a);
void print_prime_factors(double a); // Changed to void

int main()
{
    double a, b;
    cout << "Enter first number : ";
    cin >> a;
    cout << "Enter second number : ";
    cin >> b;

    int choice;

    do {
        cout << "Enter 1 for addition\n";
        cout << "Enter 2 for subtraction\n";
        cout << "Enter 3 for multiplication\n";
        cout << "Enter 4 for division\n";
        cout << "Enter 5 for exponent\n";
        cout << "Enter 6 for factorial\n";
        cout << "Enter 7 for square root\n";
        cout << "Enter 8 for square\n";
        cout << "Enter 9 for cube root\n";
        cout << "Enter 10 for cube\n";
        cout << "Enter 11 for prime checking\n";
        cout << "Enter 12 for prime factors\n";
        cout << "Enter 13 to exit\n";
        cout << "Enter your choice : ";
        cin >> choice;
        cout << "\n";

        switch (choice)
        {
            case 1:
                cout << "Addition is : " << addition(a, b) << '\n';
                break;
            case 2:
                cout << "Subtraction is : " << subtraction(a, b) << '\n';
                break;
            case 3:
                cout << "Product is : " << multiplication(a, b) << '\n';
                break;
            case 4:
                if (b == 0) cout << "Error: Division by zero\n";
                else cout << "Division is : " << division(a, b) << '\n';
                break;
            case 5:
                cout << "Exponent is : " << exponent(a, b) << '\n';
                break;
            case 6:
                cout << "Factorials are : " << factorial(a) << " " << factorial(b) << '\n';
                break;
            case 7:
                cout << "Square Roots are : " << square_root(a) << " " << square_root(b) << '\n';
                break;
            case 8:
                cout << "Squares are : " << square(a) << " " << square(b) << '\n';
                break;
            case 9:
                cout << "Cube Roots are : " << cube_root(a) << " " << cube_root(b) << '\n';
                break;
            case 10:
                cout << "Cubes are : " << cube(a) << " " << cube(b) << '\n';
                break;
            case 11:
                cout << "Is " << a << " prime? " << (isPrime(a) ? "Yes" : "No") << '\n';
                cout << "Is " << b << " prime? " << (isPrime(b) ? "Yes" : "No") << '\n';
                break;
            case 12:
                print_prime_factors(a);
                print_prime_factors(b);
                break;
            case 13:
                cout << "You have exited the program\n";
                break;
            default:
                cout << "Invalid choice\n";
                break;
        }

        cout << '\n';
    } while (choice != 13);

    return 0;
}

double addition(double a, double b) { return a + b; }
double subtraction(double a, double b) { return a - b; }
double multiplication(double a, double b) { return a * b; }
double division(double a, double b) { return a / b; }
double exponent(double a, double b) { return pow(a, b); }

double factorial(double a) {
    if (a < 0 || a != floor(a)) return -1; // Return -1 for invalid input
    double fact = 1;
    for (int i = 1; i <= static_cast<int>(a); ++i) {
        fact *= i;
    }
    return fact;
}

double square_root(double a) { return sqrt(a); }
double square(double a) { return a * a; }
double cube_root(double a) { return cbrt(a); }
double cube(double a) { return a * a * a; }

bool isPrime(double a) {
    if (a <= 1 || a != floor(a)) return false;
    int n = static_cast<int>(a);
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

void print_prime_factors(double a) {
    if (a < 2 || a != floor(a)) {
        cout << "Cannot compute prime factors for " << a << '\n';
        return;
    }

    cout << "Prime factors of " << a << " are: ";
    int n = static_cast<int>(a);
    for (int i = 2; i <= n; ++i) {
        while (n % i == 0) {
            cout << i << " ";
            n /= i;
        }
    }
    cout << '\n';
}