/* Objective function */
max: 38 T1 + 36 T2 + 49 F1 + 47 F2 + 38 C1 + 35 C2 + 68 H1 + 65 H2 - A_T - A_F - A_C - A_H;

/* Constraints */
dept1_RT: T1 + F1 <= 100000;
dept2_RT: C1 + H1 <= 90000;
dept1_OT: T2 + F2 <= 25000;
dept2_OT: C2 + H2 <= 24000;
adv_budget: A_T + A_F + A_C + A_H <= 18000;
demand_T: T1 + T2 <= 60000 + 0.72 * A_T;
demand_F: F1 + F2 <= 20000 + 0.20 * A_F;
demand_C: C1 + C2 <= 100000 + 0.80 * A_C;  /* Corrected from 0.08 to 0.80 */
demand_H: H1 + H2 <= 35000 + 0.525 * A_H; /* Corrected from 0.15 to 0.525 */
adv_T_max: A_T <= 10000;
adv_F_max: A_F <= 10000;
adv_C_max: A_C <= 10000;
adv_H_max: A_H <= 10000;

/* Variable bounds */
T1 >= 0;
T2 >= 0;
F1 >= 0;
F2 >= 0;
C1 >= 0;
C2 >= 0;
H1 >= 0;
H2 >= 0;
A_T >= 0;
A_F >= 0;
A_C >= 0;
A_H >= 0;