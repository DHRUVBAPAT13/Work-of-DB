#include <stdio.h>
#include <stdlib.h>
#define MAX 20

void enqueue(int queue[], int *front, int *rear, int value);
void dequeue(int queue[], int *front, int *rear);
void enqueueFromBeg(int queue[], int *front, int *rear, int value);
void dequeueFromEnd(int queue[], int *front, int *rear);
void display(int queue[], int front, int rear);


int main() {
    int data[MAX];
    int front = -1, rear = -1;
    int ch, value;

    do{
        printf("Queue Operations:\n");
        printf("Press 1 to Insert from end\n");
        printf("Press 2 to Insert from beginning\n");
        printf("Press 3 to Delete from beginning\n");
        printf("Press 4 to Delete from end\n");
        printf("Press 5 to Display\n");
        printf("Press 6 to Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &ch);

        switch (ch) {
            case 1:
                printf("Enter value to enqueue from end : ");
                scanf("%d", &value);
                enqueue(data, &front, &rear, value);
                break;
            case 2:
                printf("Enter value to enqueue from beginning : ");
                scanf("%d", &value);
                enqueueFromBeg(data, &front, &rear, value);
                break;
            case 3:
                dequeue(data, &front, &rear);
                break;
            case 4:
                dequeueFromEnd(data, &front, &rear);
                break;
            case 5:
                display(data, front, rear);
                break;
            case 6:
                printf("Exiting the program.\n");
                exit(0);
            default:
                printf("Invalid choice! Please try again.\n");
        }
    }while (ch != 6);
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

void dequeueFromEnd(int queue[], int *front, int *rear)
{
    if (*front == -1) 
    {
        printf("Queue UNDERFLOW\n");
        return;
    }

    printf("Dequeued value from end: %d\n", queue[*rear]);

    if (*front == *rear) 
    {
        *front = -1;
        *rear = -1;
    } 
    else 
    {
        (*rear)--;
    }
}

void enqueueFromBeg(int queue[], int *front, int *rear, int value)
{
    
    if (*rear == MAX - 1) 
    {
        printf("Queue OVERFLOW (No space to shift elements)\n");
        return;
    }

    if (*front == -1) 
    {
        *front = 0;
        *rear = 0;
        queue[*front] = value;
    } 
    else 
    {
        for (int i = *rear; i >= *front; i--) 
        {
            queue[i + 1] = queue[i];
        }
        (*rear)++;
        queue[*front] = value;
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