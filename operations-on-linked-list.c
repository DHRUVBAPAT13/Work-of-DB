#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int data;
    struct node *next;
}Node;

Node *start = NULL;

void addAtBeg(int value)
{
    Node *temp = (Node*)malloc(sizeof(Node));
    temp->data = value;

    if(start == NULL)
    {
        temp->next = NULL;
        start = temp;
    }
    else
    {
        temp->next = start;
        start = temp;
    }
}

void addAtEnd(int value)
{
    Node *temp = (Node*)malloc(sizeof(Node));
    temp->data = value;
    temp->next = NULL;

    if(start == NULL)
    {
        start = temp;
    }
    else
    {
        Node *ptr = start;
        while (ptr->next != NULL)
        {
            ptr = ptr->next;
        }
        ptr->next = temp;
    }
}

void addAtPosition(int value, int pos)
{
    if(start == NULL || pos <= 1)
    {
        addAtBeg(value);
        return;
    }

    Node *temp = (Node*)malloc(sizeof(Node));
    temp->data = value;

    Node *ptr = start;
    int i;
    for(i = 1; i < pos - 1 && ptr != NULL; i++)
    {
        ptr = ptr->next;
    }

    if(ptr == NULL)
    {
        addAtEnd(value);
        free(temp);
        return;
    }

    temp->next = ptr->next;
    ptr->next = temp;
}

void display()
{
    if(start == NULL)
    {
        printf("List is empty\n");
        return;
    }

    Node *ptr = start;
    while (ptr != NULL)
    {
        printf("%d\t", ptr->data);
        ptr = ptr->next;
    }
    printf("\n");
}

void deleteFromBeg()
{
    if(start == NULL)
        printf("List is empty\n");
    else
    {
        Node *temp = start;
        start = start->next;
        free(temp);
    }
}

void deleteFromEnd()
{
    if(start == NULL)
        printf("List is empty\n");
    else if(start->next == NULL)
    {
        free(start);
        start = NULL;
    }
    else
    {
        Node *ptr = start;
        while(ptr->next->next != NULL)
        {
            ptr = ptr->next;
        }
        free(ptr->next);
        ptr->next = NULL;
    }
}

void sortList()
{
    Node *p, *q;

    p = start;
    while (p != NULL)
    {
        q = p->next;
        while (q != NULL)
        {
            if(p->data > q->data)
            {
                int temp = p->data;
                p->data = q->data;
                q->data = temp;
            }
            q = q->next;
        }
        p = p->next;
    }
}

int main()
{
    addAtEnd(1);
    addAtEnd(22);
    addAtEnd(333);
    addAtEnd(4444);
    addAtBeg(5);
    display();

    deleteFromBeg();
    display();

    deleteFromEnd();
    display();

    addAtPosition(56, 2);
    display();

    sortList();
    display();

    return 0;
}