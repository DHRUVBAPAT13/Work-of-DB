#include <stdio.h>

void shellSort(int arr[], int n)
{
    int gap, i, j;
    int temp;
    for(gap=n-1;gap>=1;gap--)
    {
        for(i=0;i<n-gap;i++)
        {
            if(arr[i]>arr[i+gap])
            {
                temp = arr[i];
                arr[i] = arr[i+gap];
                arr[i+gap] = temp;
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

    shellSort(array, len);

    printf("\n\nSorted array : ");
    for(int j=0;j<len;j++)
        printf("%d ",array[j]);

    printf("\n");
    return 0;

}