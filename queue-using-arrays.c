#include <stdio.h>
#include <stdlib.h>
#define MAX 20

void enqueue(int queue[], int *front, int *rear, int value);
void dequeue(int queue[], int *front, int *rear);
void display(int queue[], int front, int rear);

int main() {
    int data[MAX];
    int front = -1, rear = -1;
    int ch, value;

    do{
        printf("Queue Operations:\n");
        printf("Press 1 to Insert\n");
        printf("Press 2 to Delete\n");
        printf("Press 3 to Display\n");
        printf("Press 4 to Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &ch);

        switch (ch) {
            case 1:
                printf("Enter value to enqueue: ");
                scanf("%d", &value);
                enqueue(data, &front, &rear, value);
                break;
            case 2:
                dequeue(data, &front, &rear);
                break;
            case 3:
                display(data, front, rear);
                break;
            case 4:
                printf("Exiting the program.\n");
                exit(0);
            default:
                printf("Invalid choice! Please try again.\n");
        }
    }while (ch != 4);
    return 0;
}

void enqueue(int queue[], int *front, int *rear, int value)
{
    if ((*front == 0 && *rear == MAX - 1) || (*front == *rear + 1)) 
    {
        printf("Queue OVERFLOW\n");
        return;
    }
    if (*front == -1) 
    {
        *front = 0;
    }
    (*rear)++;
    queue[*rear] = value;
}

void dequeue(int queue[], int *front, int *rear)
{
    if (*front == -1) 
    {
        printf("Queue UNDERFLOW\n");
        return;
    }
    printf("Dequeued value: %d\n", queue[*front]);
    if (*front == *rear) 
    {
        *front = -1;
        *rear = -1;
    } 
    else 
    {
        (*front)++;
    }
}

void display(int queue[], int front, int rear)
{
    if (front == -1 && rear == -1) 
    {
        printf("Queue is empty\n");
        return;
    }
    printf("Queue elements: ");
    for (int i = front; i <= rear; i++) 
    {
        printf("%d ", queue[i]);
    }
    printf("\n");
}