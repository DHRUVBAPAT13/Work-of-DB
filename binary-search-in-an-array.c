#include <stdio.h>

void sort(int *a, int n)
{
    int i, j, temp;
    for(i=0;i<n-1;i++)
    {
        for(j=0;j<n-i-1;j++)
        {
            if(*(a+j) > *(a+j+1))
            {
                temp = *(a+j);
                *(a+j) = *(a+j+1);
                *(a+j+1) = temp;
            }
        }
    }
}

int main()
{
    int array[100], search, c, n, position;
    printf("Enter no. of elements in array : ");
    scanf("%d",&n);

    printf("\nEnter the elements of the array : ");
    for(c=0;c<n;c++)
    {
        scanf("%d",&array[c]);
    }

    sort(array, n);
    printf("Array after sorting : ");
    for(int i=0;i<n;i++)
        printf("%d ",array[i]);


    printf("\nEnter element to be searched : ");
    scanf("%d",&search);

    int first, middle, last;

    first=0;
    last=n-1;
    middle = (first+last)/2;

    while (first<=last)
    {
        if(array[middle] < search)
        {
            first = middle+1;
        }
        else if(array[middle] == search)
        {
            printf("%d found at location %d.\n", search, middle + 1);
            return 0;
        }
        else
        {
            last = middle-1;
        }
        middle = (first+last)/2 ;
    }

    if(first > last)
        printf("Not found! %d isn't present in the list.\n", search);

    return 0;

}