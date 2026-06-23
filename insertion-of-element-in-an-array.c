#include <stdio.h>

int main() {
    
    
    int len;
    printf("Enter length of array : ");
    scanf("%d",&len);
    int a[len];
    printf("Enter array elements : ");
    for(int x=0;x<len;x++)
    {
        scanf("%d",&a[x]);
    }
    
    printf("\nArray elements are : ");
    for(int y=0;y<len;y++)
        printf("\t%d",a[y]);
        
    int num, i, pos, n=len;

    printf("\n\nenter number to be inserted and position where to insert : ");
    scanf("%d %d",&num,&pos);

    if(pos>len+1)
    {
        printf("Enter a number less than %d",len+1);
    }
    else
    {
        for(i=n;i>=pos;i--)
        {
            a[i]=a[i-1];
        }
        a[pos-1] = num;
        n++;
    
        printf("Array elements are : ");
        for(i=0;i<n;i++)
            printf("%d  ",a[i]);
        
    }
    
    
    return 0;
}