#include <stdio.h>
#include <stdlib.h> // Included for exit()

#define max 20

void push(int[], int*, int);
int pop(int[], int*);
void display(int[], int);

int main()
{
    int array[max];
    int top = -1;
    int num, d;

    // Put everything in a loop so the menu keeps appearing
    while(1) 
    {
        printf("\n--- Stack Menu ---");
        printf("\n1. Push");
        printf("\n2. Pop");
        printf("\n3. Display");
        printf("\n4. Exit");
        printf("\nEnter your choice: ");
        scanf("%d", &d);

        if (d == 1)
        {
            printf("Enter the element to push: ");
            scanf("%d", &num); // Take dynamic input instead of hardcoded 10
            push(array, &top, num);
        }
        else if(d == 2)
        {
            num = pop(array, &top);
            if(num != -1) // Only print if a valid element was popped
            {
                printf("Deleted element = %d\n", num);
            }
        }
        else if(d == 3)
        {
            display(array, top);
        }
        else if(d == 4)
        {
            printf("\nExiting program...\n");
            exit(0); // Safely exit the infinite loop
        }
        else
        {
            printf("\nInvalid choice! Please try again.\n");
        }
    }

    return 0;
}

void push(int arr[], int *top, int val)
{
    if(*top == max - 1)
    {
        printf("\nOVERFLOW: Stack is full!\n");
    }
    else
    {
        *top = *top + 1;
        arr[*top] = val;
        printf("%d pushed onto stack.\n", val);
    }
}

int pop(int arr[], int *top)
{
    if(*top == -1)
    {
        printf("\nUNDERFLOW: Stack is empty!\n");
        return -1; // Return a specific error code
    }
    else
    {
        int number = arr[*top];
        *top = *top - 1;
        return number; // Return the actual popped value
    }
}

void display(int arr[], int top)
{
    if(top == -1)
    {
        printf("\nStack is empty.\n");
    }
    else
    {
        printf("\nStack elements (Top to Bottom): ");
        for(int i = top; i >= 0; i--)
        {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
}