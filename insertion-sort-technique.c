#include <stdio.h>

/* this type of sort has best case complexity of O(n) 
and worst case complexity of O(n²) */

void insertionSort(int arr[], int n)
{
    int i, j, temp;

    for(j=1;j<n;j++)
    {
        temp = arr[j];
        i = j-1;
        while(arr[i]>temp && i>=0)
        {
            arr[i+1] = arr[i];
            i = i-1;
        }
        arr[i+1] = temp;
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

    insertionSort(array, len);

    printf("\nSorted array : ");
    for(int j=0;j<len;j++)
        printf("%d ",array[j]);

    printf("\n");
    return 0;

}