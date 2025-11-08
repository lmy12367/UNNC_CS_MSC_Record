/* Objective function */
/* Building Bridges -- lp_solve model */
/* Distances are from your table (tiles = km) */

min:12*A1 + 14*A2 + 20*A3 + 24*A4+ 9*B1 + 17*B2 + 23*B3 + 27*B4 + 18*C1 + 10*C2 + 12*C3 + 16*C4 +25*D1 + 15*D2 + 15*D3 + 15*D4 + 26*E1 + 16*E2 + 10*E3 + 10*E4;

A1 + A2 + A3 + A4 = 1;
B1 + B2 + B3 + B4 = 1;
C1 + C2 + C3 + C4 = 1;
D1 + D2 + D3 + D4 = 1;
E1 + E2 + E3 + E4 = 1;

A1 <= y1; A2 <= y2; A3 <= y3; A4 <= y4;
B1 <= y1; B2 <= y2; B3 <= y3; B4 <= y4;
C1 <= y1; C2 <= y2; C3 <= y3; C4 <= y4;
D1 <= y1; D2 <= y2; D3 <= y3; D4 <= y4;
E1 <= y1; E2 <= y2; E3 <= y3; E4 <= y4;

y1 + y2 + y3 + y4 = 2;

bin A1, A2, A3, A4,
    B1, B2, B3, B4,
    C1, C2, C3, C4,
    D1, D2, D3, D4,
    E1, E2, E3, E4,
    y1, y2, y3, y4;
