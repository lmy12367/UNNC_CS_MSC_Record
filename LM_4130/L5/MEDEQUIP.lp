/* Objective function */
min:600 x_F1_C1 + 800 x_F1_C2 + 700 x_F1_C3 + 400 x_F2_C1 + 900 x_F2_C2 + 600 x_F2_C3;

x_F1_C1 + x_F2_C1 = 300;
x_F1_C2 + x_F2_C2 = 200;
x_F1_C3 + x_F2_C3 = 400;

x_F1_C1 + x_F1_C2 + x_F1_C3 <= 400;
x_F2_C1 + x_F2_C2 + x_F2_C3 <= 500;


