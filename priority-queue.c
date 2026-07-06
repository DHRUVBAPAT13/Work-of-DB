#include <stdio.h>
#include <stdlib.h>

#define max 20

int front, rear;
int queue[max];

void enqueue(int value);
void dequeue();
void display();

int main()
{
    front = rear = -1;
    int ch, value;
    
    do{
        printf("Queue Operations:\n");
        printf("Press 1 to Add Priority value\n");
        printf("Press 2 to Delete\n");
        printf("Press 3 to Display\n");
        printf("Press 4 to Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &ch);

        switch (ch) {
            case 1:
                printf("Enter value : ");
                scanf("%d", &value);
                enqueue(value);
                break;
            case 2:
                dequeue();
                break;
            case 3:
                display();
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

void enqueue(int value)
{
    if(front == -1)
    {
        front = rear = 0;
        queue[0] = value;
    }
    else if(rear == max-1)
    {
        printf("Queue Overflow\n");
    }
    else
    {
        int i = rear;
        while(value < queue[i] && i>=0)
        {
            queue[i+1] = queue[i];
            i = i - 1;
        } 
        queue[i+1] = value;
        rear = rear + 1;
    }
}

void dequeue()
{
    if(front == -1)
    {
        printf("Queue Underflow\n");
    }
    else
    {
        if(front == rear)
        {
            front = rear = -1;
        }
        else
        {
            for(int i=0; i<rear; i++)
            {
                queue[i] = queue[i+1];
            }
            rear = rear -1;
        }
    }
}

void display()
{
    if (front == -1 && rear == -1) 
    {
        printf("Queue is empty\n");
        return;
    }
    printf("Queue elements : ");
    for (int i = 0; i <= rear; i++) 
    {
        printf("%d ", queue[i]);
    }
    printf("\n");
}