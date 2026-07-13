#include <stdio.h>
#include <stdlib.h>

void acceptMat(int mat[20][20], int r, int c) {
    printf("Enter matrix elements:\n");
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            scanf("%d", &mat[i][j]);
        }
    }
}

void displayMat(int mat[20][20], int r, int c) {
    printf("Matrix elements:\n");
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
}

void addMatrices(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2) {
    if(r1 != r2 || c1 != c2) {
        printf("Order of matrices do not match.\n");
    } else {
        int res[20][20];
        for(int i = 0; i < r1; i++) {
            for(int j = 0; j < c1; j++) {
                res[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        printf("Result of addition:\n");
        for(int i = 0; i < r1; i++) {
            for(int j = 0; j < c1; j++) {
                printf("%d ", res[i][j]);
            }
            printf("\n");
        }
    }
}

void subtractMatrices(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2) {
    if(r1 != r2 || c1 != c2) {
        printf("Order of matrices do not match.\n");
    } else {
        int res[20][20];
        for(int i = 0; i < r1; i++) {
            for(int j = 0; j < c1; j++) {
                res[i][j] = mat1[i][j] - mat2[i][j];
            }
        }
        printf("Result of subtraction:\n");
        for(int i = 0; i < r1; i++) {
            for(int j = 0; j < c1; j++) {
                printf("%d ", res[i][j]);
            }
            printf("\n");
        }
    }
}

void transpose(int mat[20][20], int r, int c) {
    int tmat[20][20];
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            tmat[j][i] = mat[i][j];
        }
    }
    printf("Transpose:\n");
    for(int i = 0; i < c; i++) {
        for(int j = 0; j < r; j++) {
            printf("%d ", tmat[i][j]);
        }
        printf("\n");
    }
}

void multpMat(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2) {
    if(c1 != r2) {
        printf("Matrices cannot be multiplied.\n");
        return;
    }
    int res[20][20] = {0};
    for(int i=0; i<r1; i++) {
        for(int j=0; j<c2; j++) {
            for (int k = 0; k<c1; k++) {
                res[i][j] += mat1[i][k]*mat2[k][j];
            }
        }
    }
    printf("Result of multiplication:\n");
    for(int i = 0; i < r1; i++) {
        for(int j = 0; j < c2; j++) {
            printf("%d ", res[i][j]);
        }
        printf("\n");
    }
}

int sparseCheck(int r, int c, int a[20][20]) {
    int n=0, z=0;
    for(int i=0;i<r;i++) {
        for(int j=0;j<c;j++) {
            if(a[i][j] != 0) n++;
            else z++;
        }
    }
    return (n < z);
}

void compactFrom(int r, int c, int a[20][20]) {
    int b[401][3];
    b[0][0] = r;
    b[0][1] = c;
    int k = 1;
    for(int i=0;i<r;i++) {
        for(int j=0;j<c;j++) {
            if(a[i][j] != 0) {
                b[k][0] = i;
                b[k][1] = j;
                b[k][2] = a[i][j];
                k++;
            }
        }
    }
    b[0][2] = k-1;
    printf("\nThe compact form is:\n");
    for(int i=0;i<k;i++) {
        for(int j=0;j<3;j++) {
            printf("%d ",b[i][j]);
        }
        printf("\n");
    }  
}

int main() {   
    int mat1[20][20], mat2[20][20];
    int r1, c1, r2, c2;

    printf("Enter order of first matrix:\n");
    if (scanf("%d %d", &r1, &c1) != 2 || r1 < 1 || r1 > 20 || c1 < 1 || c1 > 20) {
        printf("Invalid order for first matrix. Use values between 1 and 20.\n");
        return 1;
    }
    acceptMat(mat1, r1, c1);
    displayMat(mat1, r1, c1);
    
    printf("Enter order of second matrix:\n");
    if (scanf("%d %d", &r2, &c2) != 2 || r2 < 1 || r2 > 20 || c2 < 1 || c2 > 20) {
        printf("Invalid order for second matrix. Use values between 1 and 20.\n");
        return 1;
    }
    acceptMat(mat2, r2, c2);
    displayMat(mat2, r2, c2);

    int choice, ch;
    while (1) {
        printf("Enter 1 for addition\nEnter 2 for subtraction\nEnter 3 for transpose\nEnter 4 for multiplication\nEnter 5 for sparse matrix\n\n");
        if (scanf("%d", &choice) != 1) {
            int ch2;
            while ((ch2 = getchar()) != '\n' && ch2 != EOF);
            printf("Invalid input. Please enter a number between 1 and 5.\n");
            continue;
        }
        if (choice >= 1 && choice <= 5) {
            break;
        }
        printf("Invalid choice. Please enter a number between 1 and 5.\n");
    }

    switch(choice) {
        case 1:
            addMatrices(mat1, mat2, r1, c1, r2, c2);
            break;
        case 2:
            subtractMatrices(mat1, mat2, r1, c1, r2, c2);
            break;
        case 3:
            printf("Enter 1 for matrix 1 or 2 for matrix 2: ");
            scanf("%d", &ch);
            if(ch == 1) transpose(mat1, r1, c1);
            else if(ch == 2) transpose(mat2, r2, c2);
            else printf("Invalid matrix number.\n");
            break;
        case 4:
            multpMat(mat1, mat2, r1, c1, r2, c2);
            break;
        case 5:
            printf("Enter 1 for matrix 1 or 2 for matrix 2: ");
            scanf("%d", &ch);
            if(ch == 1) {
                if(sparseCheck(r1, c1, mat1)) {
                    printf("It is a sparse matrix\n");
                    compactFrom(r1, c1, mat1);
                } else {
                    printf("It is not a sparse matrix\n");
                }
            } else if(ch == 2) {
                if(sparseCheck(r2, c2, mat2)) {
                    printf("It is a sparse matrix\n");
                    compactFrom(r2, c2, mat2);
                } else {
                    printf("It is not a sparse matrix\n");
                }
            } else {
                printf("Invalid matrix number\n");
            }
            break;
    }
    return 0;
}

