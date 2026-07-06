#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int data;
    struct node *next;
}Node;

Node *front = NULL, *rear = NULL;

void enqueueAtEnd(int val)
{
    if(front == NULL)
    {
        front = (Node*)malloc(sizeof(Node));
        front ->data = val;
        front ->next = NULL;
        rear = front;
    }
    else
    {
        Node *temp = (Node*)malloc(sizeof(Node));
        temp ->data = val;
        temp ->next = NULL;
        rear ->next = temp;
        rear = temp;
    }
}

void enqueueAtBeg(int val)
{
    if(front == NULL)
    {
        front = (Node*)malloc(sizeof(Node));
        front ->data = val;
        front ->next = NULL;
        rear = front;
    }
    else
    {
        Node *temp = (Node*)malloc(sizeof(Node));
        temp ->data = val;
        temp ->next = front;
        front = temp;
    }
}

void dequeueFromBeg()
{
    
    if(front == NULL) 
    {
        printf("Queue is empty\n");
    }
    else if(front == rear)
    {
        free(front);
        front = rear = NULL;
    }
    else
    {
        Node *temp;
        temp = front;
        front = front ->next;
        free(temp);
    }
}

void dequeueFromEnd()
{
    
    if(front == NULL) 
    {
        printf("Queue is empty\n");
    }
    else if(front == rear)
    {
        free(front);
        front = rear = NULL;
    }
    else
    {
        Node *temp;
        temp = front;
        while (temp ->next != rear)
        {
            temp = temp ->next;
        }
        free(rear);
        temp ->next = NULL;
        rear = temp;
    }
}

void display()
{
    if(front == NULL)
    {
        printf("Queue is empty\n");
        return;
    }
    
    Node *ptr = front;
    printf("Queue elements: ");
    while (ptr != NULL)
    {
       printf("%d ", ptr ->data);
        ptr = ptr ->next;
    }
    printf("\n");
}

int main() {

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
                enqueueAtEnd(value);
                break;
            case 2:
                printf("Enter value to enqueue from beginning : ");
                scanf("%d", &value);
                enqueueAtBeg(value);
                break;
            case 3:
                dequeueFromBeg();
                break;
            case 4:
                dequeueFromEnd();
                break;
            case 5:
                display();
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