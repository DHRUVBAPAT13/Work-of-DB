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

        while(ptr->next != start)
            ptr = ptr -> next;

        temp = (Node*)malloc(sizeof(Node));
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
    if(start == NULL)
    {
        start = temp;
        start -> next = start;
    }
    else
    {
        ptr = start;
        while (ptr->next != start)
        {
            ptr = ptr -> next;
        }
        ptr -> next = temp;
        temp -> next = start;
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
        if(ptr->next == NULL)
        {
            break;
        }
        ptr = ptr->next;
    }
    temp->next = ptr->next;
    ptr->next = temp;
}

void display()
{
    if(start == NULL) return;
    Node *ptr;
    ptr = start;
    do
    {
        printf("%d\t",ptr->data);
        ptr = ptr->next;
    } while (ptr!=start);
    printf("\n");
}

int main()
{
    addAtEnd(1);
    addAtEnd(22);
    addAtEnd(333);
    addAtEnd(4444);
    addAtBeg(5);
    display();

    return 0;
}