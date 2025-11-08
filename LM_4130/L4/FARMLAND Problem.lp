/* Objective function */
max:1600x1+250x2+480x3+500x4 ;

/* Constraints */
0.2x2+0.2x3+0.2x4<=4x1;
x4<=0.3x2+0.3x3+0.3x4;
x2<=2000-2x2-2x3-2x4;
4x1+x2+x3+x4<=1000;
40x1+10x2+12x3+25x4<=12000;

/* Variable bounds */
x1>=0;
x2>=0;
x3>=0;
x4>=0;

/* Integer declaration */
int x1;