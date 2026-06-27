#include <stdio.h>
#include <stdlib.h>

//uses a variable named key

struct node
{
    int data;
    int key;
    struct node *next; 
};

struct node *head = NULL;

void insertFirst(int key, int data)
{
    struct node *link = (struct node*)malloc(sizeof(struct node));

    link -> key = key;
    link -> data = data;

    link -> next = head;

    head = link;
}

int main()
{

    insertFirst(1, 10);

    insertFirst(2, 20);

    insertFirst(3, 30);

    insertFirst(4, 40);

    struct node *ptr = head;
    printf("Linked List Elements: \n");
    while(ptr != NULL) 
    {
        printf("[%d, %d] -> ", ptr->key, ptr->data);
        ptr = ptr->next; 
    }
    printf("NULL\n");

    struct node *ptr1 = head;
    printf("Head points to address: %p\n\n", (void*)head);
    
    printf("Linked List Elements with Memory Addresses: \n");
    while(ptr1 != NULL) {
        // %p prints the pointer's memory address
        printf("Node Location: %p | Key: %d, Data: %d | Next Node Location: %p\n", 
               (void*)ptr1, ptr1->key, ptr1->data, (void*)ptr1->next);
        
        ptr1 = ptr1->next; // Move to the next node
    }

    return 0;
}









