/* 目标函数 */
max: 21.25 xwoo + 15 xmoo + 18 xwoc + 12 xmoc + 15 xnoc + 17.1 xwot + 14.4 xmot + 10.8 xnot;

/* 原料上限 */
xwoo + xwoc + xwot <= 300;
xmoo + xmoc + xmot <= 600;
xnoc + xnot <= 400;

/* 产品上限 */
0.85 xwoo + 0.6 xmoo <= 150;
0.9 xwoc + 0.6 xmoc + 0.75 xnoc <= 750;
0.95 xwot + 0.8 xmot + 0.6 xnot <= 550;


/* 非负 */
xwoo >= 0;
xwoc >= 0;
xwot >= 0;
xmoo >= 0;
xmoc >= 0;
xmot >= 0;
xnoc >= 0;
xnot >= 0;