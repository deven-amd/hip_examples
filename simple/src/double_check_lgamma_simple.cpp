#include <stdio.h>
#include <math.h>

int main() {

  for (int i=0; i<6; i++) {
    double val = -2.5 + 1.0*i;
    printf("lgamma(%f) = %f\n", val, lgamma(val));
  }

  return 0;
}
