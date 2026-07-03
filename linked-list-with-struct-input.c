#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct student
{
    int rollno;
    char name[30];
    float cgpa;
}Student;

typedef struct node{
    Student data;
    struct node *next;
}Node;

Node *start = NULL;

void addAtBeg(Student value)
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

void addAtEnd(Student value)
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

void addAtPosition(Student value, int pos)
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
    printf("\n");
    if(start == NULL)
    {
        printf("List is empty\n");
        return;
    }

    Node *ptr = start;
    while (ptr != NULL)
    {
        printf("%d %s %.2f\n", ptr->data.rollno,ptr->data.name,ptr->data.cgpa);
        ptr = ptr->next;
    }
    printf("\n");
}

int main()
{
    Student val;

    for (int i = 0; i < 5; i++)
    {
        printf("\nEnter student details : ");
        scanf("%d %s %f",&val.rollno,val.name,&val.cgpa);
        addAtEnd(val);
    }

    display();

    return 0;
}