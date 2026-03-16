extern int my_func(int);

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int a = atoi(argv[1]);   // convert command-line string to int
    printf("Result of execution for (%d): %d\n", a, my_func(a));
}
