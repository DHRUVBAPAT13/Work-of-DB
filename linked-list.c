#include <stdio.h>
#include <stdlib.h>

// 1. Define the node structure (Now without the key variable)
struct node
{
    int data;
    struct node *next; 
};

struct node *head = NULL;

// 2. Updated function (Only takes data as an argument now)
void insertFirst(int data)
{
    struct node *link = (struct node*)malloc(sizeof(struct node));

    link -> data = data;
    link -> next = head;

    head = link;
}

int main()
{
    // Inserting values without needing a key
    insertFirst(10);
    insertFirst(20);
    insertFirst(30);
    insertFirst(40);

    // Printing simple elements
    struct node *ptr = head;
    printf("Linked List Elements: \n");
    while(ptr != NULL) 
    {
        printf("[%d] -> ", ptr->data);
        ptr = ptr->next; 
    }
    printf("NULL\n\n");

    // Printing elements with memory addresses
    struct node *ptr1 = head;
    printf("Head points to address: %p\n\n", (void*)head);
    
    printf("Linked List Elements with Memory Addresses: \n");
    while(ptr1 != NULL) {
        printf("Node Location: %p | Data: %d | Next Node Location: %p\n", 
               (void*)ptr1, ptr1->data, (void*)ptr1->next);
        
        ptr1 = ptr1->next; 
    }

    return 0;
}