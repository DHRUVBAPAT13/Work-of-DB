#include <stdio.h>
#include "utils.h" // Notice quotes instead of < > for local files!

int main() {
    int result = add_numbers(10, 20);
    printf("The result of the module call is: %d\n", result);
    return 0;
}