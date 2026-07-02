#include <stdio.h>
#include <stdlib.h>

typedef struct node
{
    int data;
    struct node *prev, *next;
}Node;

Node *start, *end;

void addAtEnd(int value)
{
    if(start == NULL)
    {
        start = (Node*)malloc(sizeof(Node));
        start ->data = value;
        start ->next = NULL;
        start ->prev = NULL;
        end = start;
    }
    else
    {
        Node *temp;
        temp = (Node*)malloc(sizeof(Node));
        temp ->data = value;
        temp ->next = NULL;
        temp ->prev = end;
        end ->next = temp;
        end = temp;
    }
}

void display()
{
    Node *ptr;
    if(start == NULL)
        printf("List is empty.\n");
    else
    {
        ptr = start;
        while (ptr != NULL)
        {
            printf("%d ",ptr->data);
            ptr = ptr->next;
        }
    }
    printf("\n");
}

void addAtBeg(int value)
{
    if(start == NULL)
    {
        start = (Node*)malloc(sizeof(Node));
        start ->data = value;
        start ->next = NULL;
        start ->prev = NULL;
        end = start;
    }
    else
    {
        Node *temp;
        temp = (Node*)malloc(sizeof(Node));
        temp ->data = value;
        temp ->next = start;
        temp ->prev = NULL;
        start ->prev = temp;
        start = temp;
    }
}

void addAtPosition(int value, int pos)
{
    if(start == NULL || pos <= 1)
    {
        addAtBeg(value);
        return;
    }

    Node *ptr = start;
    int i;
    for(i = 1; i < pos - 1 && ptr != NULL; i++)
    {
        ptr = ptr->next;
    }

    if(ptr == NULL)
    {
        addAtEnd(value);
        return;
    }

    Node *temp = (Node*)malloc(sizeof(Node));
    temp->data = value;
    temp->next = ptr->next;
    temp->prev = ptr;
    ptr->next = temp;

    if(temp->next != NULL)
    {
        temp->next->prev = temp;
    }
    else
    {
        end = temp;
    }
}

int main()
{

    start = end = NULL;


    for(int i=1;i<=5;i++)
    {
        addAtBeg(i*10);
    }
    display();

    addAtPosition(25, 3);
    display();

    return 0;
}