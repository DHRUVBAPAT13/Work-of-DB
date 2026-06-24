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
    
    printf("\nInitial Array elements are : ");
    for(int y=0;y<len;y++)
        printf("  %d",a[y]);
        
    int  i, pos, n=len;

    printf("\n\nenter position of element to be deleted : ");
    scanf("%d",&pos);

    if(pos>len-1)
    {
        printf("Enter a number less than %d",len-1);
    }
    else
    {
        for(i=pos-1;i<n-1;i++)
                a[i]=a[i+1];
    
        a[n-1] = 0;
        n--;
    
        printf("After deletion array is : ");
        for(i=0;i<n;i++)
            printf("%d  ",a[i]);
        
    }
    printf("\n");
    return 0;
}