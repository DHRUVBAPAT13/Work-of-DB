#include <stdio.h>

// time complexity - O(n²)

void selectionSort(int arr[], int n)
{
    int i, j;
    int temp;
    for(i=0;i<n-1;i++)
    {
        for(j=i+1;j<n;j++)
        {
            if(arr[i]>arr[j])
            {
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

int main()
{
    int len;
    printf("Enter length of array : ");
    scanf("%d",&len);

    int array[len];
    printf("\nEnter array elements : ");
    for(int i=0;i<len;i++)
        scanf("%d",&array[i]);

    printf("Original array is : ");
    for(int i=0;i<len;i++)
        printf("%d ",array[i]);

    selectionSort(array, len);

    printf("\nSorted array is : ");
    for(int i=0;i<len;i++)
        printf("%d ",array[i]);
    printf("\n");
    return 0;
}
