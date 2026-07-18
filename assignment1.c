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

void addMatrices(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2, int res[20][20]) {
    if(r1 != r2 || c1 != c2) {
        printf("Order of matrices do not match.\n");
        return 0;
    } 
    else {
        for(int i = 0; i < r1; i++) {
            for(int j = 0; j < c1; j++) {
                res[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
    }
}
int subtractMatrices(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2, int res[20][20]) {
    if(r1 != r2 || c1 != c2) {
        printf("Order of matrices do not match.\n");
        return 0;
    } 
    for(int i = 0; i < r1; i++) {
        for(int j = 0; j < c1; j++) {
            res[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return 1;
}

void transpose(int mat[20][20], int r, int c, int tmat[20][20]) {
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            tmat[j][i] = mat[i][j];
        }
    }
}

int multpMat(int mat1[20][20], int mat2[20][20], int r1, int c1, int r2, int c2, int res[20][20]) {
    if(c1 != r2) {
        
        printf("Matrices cannot be multiplied.\n");
        return 0;
    }
    for(int i = 0; i < r1; i++) {
        for(int j = 0; j < c2; j++) {
            res[i][j] = 0;
            for (int k = 0; k < c1; k++) {
                res[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return 1;
}

int sparseCheck(int r, int c, int a[20][20]) {
    int nonZero = 0, zero = 0;
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            if(a[i][j] != 0) nonZero++;
            else zero++;
        }
    }
    return (nonZero < zero);
}

void compactForm(int r, int c, int a[20][20], int b[401][3]) {
    b[0][0] = r;
    b[0][1] = c;
    int k = 1;
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            if(a[i][j] != 0) {
                b[k][0] = i;
                b[k][1] = j;
                b[k][2] = a[i][j];
                k++;
            }
        }
    }
    b[0][2] = k - 1;
}

int main() {   
    int mat1[20][20], mat2[20][20];
    int r1, c1, r2, c2;

    printf("Enter order of first matrix:\n");
    if (scanf("%d %d", &r1, &c1) != 2 || r1 < 1 || r1 > 20 || c1 < 1 || c1 > 20) {
        return 1;
    }
    acceptMat(mat1, r1, c1);
    displayMat(mat1, r1, c1);
    
    printf("Enter order of second matrix:\n");
    if (scanf("%d %d", &r2, &c2) != 2 || r2 < 1 || r2 > 20 || c2 < 1 || c2 > 20) {
        return 1;
    }
    acceptMat(mat2, r2, c2);
    displayMat(mat2, r2, c2);

    int choice;
    printf("Enter 1 for addition\nEnter 2 for subtraction\nEnter 3 for transpose\nEnter 4 for multiplication\nEnter 5 for sparse matrix\n\n");
    scanf("%d", &choice);


    switch(choice) {
        case 1: {
            int res[20][20];
            if (addMatrices(mat1, mat2, r1, c1, r2, c2, res)) {
                displayMat(res, r1, c1);
            }
            break;
        }
        case 2: {
            int res[20][20];
            if (subtractMatrices(mat1, mat2, r1, c1, r2, c2, res)) {
                displayMat(res, r1, c1);
            }
            break;
        }
        case 3: {
            int ch, res[20][20];
            scanf("%d", &ch);
            if(ch == 1) {
                transpose(mat1, r1, c1, res);
                displayMat(res, c1, r1);
            } else if(ch == 2) {
                transpose(mat2, r2, c2, res);
                displayMat(res, c2, r2);
            }
            break;
        }
        case 4: {
            int res[20][20];
            if (multpMat(mat1, mat2, r1, c1, r2, c2, res)) {
                displayMat(res, r1, c2);
            }
            break;
        }
        case 5: {
            int ch;
            scanf("%d", &ch);
            int compactRes[401][3]; 
            
            if(ch == 1) {
                if(sparseCheck(r1, c1, mat1)) 
                {
                    printf("It is a sparse matrix\n");
                    compactForm(r1, c1, mat1, compactRes);
                    for(int i = 0; i <= compactRes[0][2]; i++) 
                    {
                        printf("%d %d %d\n", compactRes[i][0], compactRes[i][1], compactRes[i][2]);
                    }
                }
                else 
                {
                    printf("It is not a sparse matrix\n");
                }
            } 
            else if(ch == 2) {
                if(sparseCheck(r2, c2, mat2)) 
                {
                    printf("It is a sparse matrix\n");
                    compactForm(r2, c2, mat2, compactRes);
                    for(int i = 0; i <= compactRes[0][2]; i++) {
                        printf("%d %d %d\n", compactRes[i][0], compactRes[i][1], compactRes[i][2]);
                    }
                }
                else 
                {
                    printf("It is not a sparse matrix\n");
                }
            }
            break;
        }
    }
    return 0;
}