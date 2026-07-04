#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int data;
    struct node *next;
}Node;

Node *top = NULL;

void push(int value)
{
    if(top == NULL)
    {
        top = (Node*)malloc(sizeof(Node));
        top ->data = value;
        top ->next = NULL;
    }
    else
    {
        Node *ptr;
        ptr = (Node*)malloc(sizeof(Node));
        ptr ->next = top;
        ptr ->data = value;
        top = ptr;
    }
}

void pop()
{
    if(top == NULL)
        printf("\nStack UNDERFLOW");
    else
    {
        Node *ptr = top;
        top = top->next;
        free(ptr);
    }
}

void display()
{
    if(top == NULL)
    {
        printf("List is empty\n");
        return;
    }

    Node *ptr = top;
    while (ptr != NULL)
    {
        printf("%d\t", ptr->data);
        ptr = ptr->next;
    }
    printf("\n");
}

int main()
{
    int num;
    for (int i = 0; i < 5; i++)
    {
        printf("Enter a number : ");
        scanf("%d",&num);
        push(num);
    }

    display();

    pop();
    display();

    return 0;
}