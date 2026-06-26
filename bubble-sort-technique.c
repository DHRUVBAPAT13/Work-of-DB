#include <stdio.h>

/* this uses pointers,
 if pointers are unwanted *(arr+j) = arr[j] 
 Also if we change the comparison operator, 
 we can get ascending or descending order.*/

void bubbleSort(int arr[], int n)
{
    int i, j, temp;
    for(i=0;i<n-1;i++)
    {
        for(j=0;j<n-i-1;j++)
        {
            if(*(arr+j) > *(arr+j+1))
            {
                temp = *(arr+j);
                *(arr+j) = *(arr+j+1);
                *(arr+j+1) = temp;
            }
        }
    }
}


int main()
{
    int len;
    printf("Enter length of the array : ");
    scanf("%d",&len);

    int array[len];
    printf("Enter array elements : ");
    for(int i=0;i<len;i++)
        scanf("%d",&array[i]);

    printf("\nInitial array : ");

    for(int i=0;i<len;i++)
        printf("%d ",array[i]);

    bubbleSort(array, len);

    printf("\nSorted array : ");
    for(int j=0;j<len;j++)
        printf("%d ",array[j]);

    printf("\n");
    return 0;

}