#include <stdio.h>
#include <stdlib.h>

#define max 10

void compactFrom(int r, int c, int a[r][c])
{
    int b[max][3];

    b[0][0] = r;
    b[0][1] = c;
    int k = 1;

    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if(a[i][j] != 0)
            {
                b[k][0] = i;
                b[k][1] = j;
                b[k][2] = a[i][j];
                k++;
            }
        }
    }
    b[0][2] = k-1;
    
    printf("\nThe compact form is : \n");
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<3;j++)
        {
            printf("%d ",b[i][j]);
        }
        printf("\n");
    }  
}

int sparseCheck(int r, int c, int a[r][c])
{
    int n=0, z=0;
    int s;
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            if(a[i][j] != 0)
                n++;
            else
                z++;
        }
    }
    if(n<z)
        s=1;
    else
        s=0;

    return s;
}

int main()
{
    int rlen, clen;
    printf("Enter rows and columns of matrix : ");
    scanf("%d %d",&rlen,&clen);

    if(rlen>max || clen>max)
    {
        printf("Enter smaller values for rows and columns.\n");
        exit(0);
    }

    int array[rlen][clen];

    printf("Enter matrix elements : ");
    for(int i=0;i<rlen;i++)
    {
        for(int j=0;j<clen;j++)
        {
            scanf("%d",&array[i][j]);
        }
    }

    printf("\nMatrix is :\n");
    for(int i=0;i<rlen;i++)
    {
        for(int j=0;j<clen;j++)
        {
            printf("%d ",array[i][j]);
        }
        printf("\n");
    }   
    
    int check = sparseCheck(rlen, clen, array);

    if(check == 1)
    {
        compactFrom(rlen, clen, array);
    }
    else
    {
        printf("\nNot a sparse matrix.");
    }
    printf("\n");
    return 0;
    
}