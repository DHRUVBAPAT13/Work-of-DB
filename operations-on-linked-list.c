#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int data;
    struct node *next;
}Node;

Node *start;


void addAtBeg(int value)
{
    if(start == NULL)
    {
        start = (Node*)malloc(sizeof(Node));
        start -> data = value;
        start -> next = start;
    }
    else
    {
        Node *ptr, *temp;
        ptr = start;
        while(ptr -> next != start)
        {
            ptr = ptr -> next;
        }
        temp = (Node*)malloc(sizeof(temp));
        temp -> data = value;
        temp -> next = start;
        start = temp;
        ptr -> next = start;
    }
}
void addAtEnd(int value)
{
    Node *temp, *ptr;
    temp = (Node*)malloc(sizeof(Node));
    temp -> data = value;
    temp -> next = NULL;
    if(start == NULL)
    {
        start = temp;
    }
    else
    {
        ptr = start;
        while (ptr->next != NULL)
        {
            ptr = ptr -> next;
        }
        ptr -> next = temp;
    }
}

void addAtPosition(int value, int pos)
{
    Node *temp, *ptr;
    temp = (Node*)malloc(sizeof(Node));
    temp -> data = value;
    ptr = start;
    for(int i=2;i<pos;i++)
    {
        if(ptr->next = NULL)
        {
            break;
        }
        ptr = ptr->next;
    }
    temp->next = ptr->next;
    ptr->next = start;
}

void display()
{
    Node *ptr;
    ptr = start;
    while (ptr!=NULL)
    {
        printf("%d\n",ptr->data);
        ptr = ptr->next;
    }
}

int main()
{
    addAtEnd(1);
    addAtEnd(22);
    addAtEnd(333);
    addAtEnd(4444);
    display();

    return 0;
}