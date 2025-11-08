/* Objective function */
min:46000*Amy + 43000*Bill + 47000*Carl + 36000*Dan + 43000*Eva + 55000*Fred + 68000*Greg + 64000*Henry + 60000*Ida;

3*Amy + 1*Bill + 3*Carl + 4*Dan + 4*Eva + 1*Fred + 3*Greg + 5*Henry + 3*Ida >= 15;
5*Amy + 2*Bill + 4*Carl + 3*Dan + 2*Eva + 3*Fred + 1*Greg + 4*Henry + 3*Ida >= 15;
1*Amy + 5*Bill + 2*Carl + 1*Dan + 2*Eva + 4*Fred + 5*Greg + 2*Henry + 3*Ida >= 15;
/* Variable bounds */
bin Amy, Bill, Carl, Dan, Eva, Fred, Greg, Henry, Ida;