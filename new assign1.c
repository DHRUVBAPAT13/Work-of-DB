#include <stdio.h>

void acceptMat(int mat[20][20], int r, int c)
{
    printf("Enter matrix elements:\n");
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            scanf("%d", &mat[i][j]);
        }
    }
}

void displayMat(int mat[20][20], int r, int c)
{
    printf("Matrix elements :\n");
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
}

void addMatrices(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2)
{
    if(r1 != r2 || c1 != c2)
    {
        printf("Order of matrices do not match.\n");
    }
    else
    {
        int res[20][20];
        for(int i = 0; i < r1; i++)
        {
            for(int j = 0; j < c1; j++)
            {
                res[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        displayMat(res, r1, c1);
    }
}

void subtractMatrices(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2)
{
    if(r1 != r2 || c1 != c2)
    {
        printf("Order of matrices do not match.\n");
    }
    else
    {
        int tmat[20][20];
        for(int i = 0; i < r1; i++)
        {
            for(int j = 0; j < c1; j++)
            {
                tmat[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
        displayMat(tmat, r1, c1);
    }
}

void transpose(int mat[20][20], int r, int c)
{
    int res[20][20];
    for(int i = 0; i < r; i++)
    {
        for(int j = 0; j < c; j++)
        {
            res[i][j] = mat[j][i];
        }
        displayMat(res, c, r);
    }
}

void multpMat(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2)
{
    if(c1 != r2)
    {
        printf("Matrices cannot be multiplied.\n");
    }

    int res[r1][c2];

    for(int i=0; i<r1; i++)
    {
        for(int j=0; j<c2; j++)
        {
            res[i][j] = 0;
        }
    }

    for(int i=0; i<r1; i++)
    {
        for(int j=0; j<c2; j++)
        {
            for (int k = 0; k<c1; k++)
            {
                res[i][j] =+ mat1[i][k]*mat2[k][j];
            }
            
        }
    }
    displayMat(res, r1, c2);

}

int main()
{   
    int mat1[20][20], mat2[20][20];
    int r1, c1, r2, c2;

    printf("Enter order of first matrix:\n");
    scanf("%d %d", &r1, &c1);  
    acceptMat(mat1, r1, c1);
    displayMat(mat1, r1, c1);
    
    printf("Enter order of second matrix:\n");
    scanf("%d %d", &r2, &c2);
    acceptMat(mat2, r2, c2);
    displayMat(mat2, r2, c2);

    int choice, ch;
    printf("Enter 1 for addition\nEnter 2 for subtraction\nEnter 3 for transpose\nEnter 4 for multiplication\n");
    scanf("%d", &choice);

    switch(choice)
    {
        case 1:
            addMatrices(mat1, mat2, r1, c1, r2, c2);
            break;
        case 2:
            subtractMatrices(mat1, mat2, r1, c1, r2, c2);
            break;
        case 3:
            printf("enter 1 for matrix 1 or 2 for matrix 2 : ");
            scanf("%d", &ch);
            if(ch == 1)
            {
                transpose(mat1, r1, c1);
            }
            else if(ch == 2)
            {
                transpose(mat2, r2, c2);
            }
            break;
            
        case 4:
            multpMat(mat1, mat2, r1, c1, r2, c2);
            break;
            
        default:
            printf("Invalid choice.\n");
            break;
    }
    return 0;
}


                
    
