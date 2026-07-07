#include <stdio.h>
#define max 20

void sortArray(int arr[], int n)
{
    //this is insertion sort function
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
    int array1[max], array2[max];
    int len1, len2;
    int m_array[max*2];

    printf("Enter length of first array : ");
    scanf("%d",&len1);

    printf("Enter elements of first array : ");
    for(int i=0; i<len1; i++)
    {
        scanf("%d",&array1[i]);
    }

    printf("Enter length of second array : ");
    scanf("%d",&len2);

    printf("Enter elements of second array : ");
    for(int i=0; i<len2; i++)
    {
        scanf("%d",&array2[i]);
    }

    printf("The arrays are : \n");
    for(int i=0; i<len1; i++)
    {
        printf("%d ",array1[i]);
    }
    printf("\n");
    for(int i=0; i<len2; i++)
    {
        printf("%d ",array2[i]);
    }

    int i, j;
    for(i=0,j=0;i<len1,j<len2;i++,j++)
    {
        m_array[i] = array1[i];
        m_array[j+len1] = array2[j];
    }

    int k = len1 +len2;

    sortArray(m_array, k);

    printf("\nThe sorted array is : \n");

    for(int x=0; x<k; x++)
    {
        printf("%d ",m_array[x]);
    }
    
    return 0;
}