#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int data;
    struct node *next;
}Node;

Node *front = NULL, *rear = NULL;

// Correct Enqueue: Insert at the REAR
void enqueue(int val)
{
    Node *newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        printf("Memory allocation failed!\n");
        return;
    }
    newNode->data = val;
    newNode->next = NULL;

    // If queue is empty, both front and rear point to the new node
    if (front == NULL) {
        front = rear = newNode;
    } 
    // Otherwise, link it to the end and move the rear pointer
    else {
        rear->next = newNode;
        rear = newNode;
    }
}

// Correct Dequeue: Remove from the FRONT
void dequeue()
{
    // Case 1: Queue is empty
    if(front == NULL) {
        printf("Queue is empty\n");
        return;
    }
    
    Node *temp = front;
    printf("%d dequeued successfully.\n", temp->data);

    // Move front to the next node
    front = front->next;

    // Case 2: If the queue became empty after deletion, update rear as well
    if (front == NULL) {
        rear = NULL;
    }

    free(temp);
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
        printf("%d  ", ptr->data);
        ptr = ptr->next;
    }
    printf("\n");
}

int main()
{
    int ch, value;

    do
    {
        printf("Queue Operations:\n");
        printf("Press 1 to Insert (Enqueue)\n");
        printf("Press 2 to Delete (Dequeue)\n");
        printf("Press 3 to Display\n");
        printf("Press 4 to Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &ch);

        switch (ch) 
        {
            case 1:
                printf("Enter value to enqueue: ");
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
    } while (ch != 4);
    
    return 0;
}