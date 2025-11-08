/* Objective function */
max: 0.95 xas + 1.15 xaj - yaj - yas;

/* Constraints */
0.8 xas + 0.6 xaj + yaj + yas <= 16000;
xas - 3 yas <= 5000;
xaj - 5 yaj <= 4000;
0.4 xaj - 0.6 xas <= 0;
0.8 xaj - 0.2 xas >= 0;

/* Variable bounds */
xas >= 0;
xaj >= 0;
yas >= 0;
yaj >= 0;
int xas;
int xaj;
int yas;
int yaj;