/* Objective function */

min:0.60 x_P1_W1 + 0.56 x_P1_W2 + 0.22 x_P1_W3 + 0.40 x_P1_W4
+ 0.36 x_P2_W1 + 0.30 x_P2_W2 + 0.28 x_P2_W3 + 0.58 x_P2_W4 +
0.65 x_P3_W1 + 0.68 x_P3_W2 + 0.55 x_P3_W3 + 0.42 x_P3_W4
+ 0 x_P1_Dum + 0 x_P2_Dum + 0 x_P3_Dum;

x_P1_W1 + x_P2_W1 + x_P3_W1 =  7500;
x_P1_W2 + x_P2_W2 + x_P3_W2 =  8500;
x_P1_W3 + x_P2_W3 + x_P3_W3 =  9500;
x_P1_W4 + x_P2_W4 + x_P3_W4 =  8000;
x_P1_Dum + x_P2_Dum + x_P3_Dum = 500;

x_P1_W1 + x_P1_W2 + x_P1_W3 + x_P1_W4 + x_P1_Dum <=  9000;
x_P2_W1 + x_P2_W2 + x_P2_W3 + x_P2_W4 + x_P2_Dum <= 12000;
x_P3_W1 + x_P3_W2 + x_P3_W3 + x_P3_W4 + x_P3_Dum <= 13000;

