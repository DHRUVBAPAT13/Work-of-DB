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

void addAtPos(int pos, int value)
{
    Node *ptr;
}

int main()
{

    start = end = NULL;


    for(int i=1;i<=5;i++)
    {
        addAtBeg(i*10);
    }
    display();

    return 0;
}