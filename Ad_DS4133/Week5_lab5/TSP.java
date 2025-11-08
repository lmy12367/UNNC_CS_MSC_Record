import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;

public class TSP {
    static final int N_CTTIES=4;
    static final String[] CITY_NAMES={"A","B","C","D"};

    static final int [][] DIST={
            {0,2,3,1},
            {2,0,4,5},
            {3,4,0,2},
            {1,5,2,0}
    };

    static int calcDistance(int [] tour){
        int total=0;
        for(int i=0;i<tour.length-1;i++){
            total += DIST[tour[i]][tour[i+1]];
        }

        total += DIST[tour[tour.length-1]][tour[0]];
        return total;
    }

    static String formatTour(int[] tour){
        StringBuilder sb=new StringBuilder();
        for(int i=0;i<tour.length;i++){
            sb.append(CITY_NAMES[tour[i]]);
            if(i<tour.length-1){
                sb.append("->");
            }
        }
        sb.append("->").append(CITY_NAMES[tour[0]]);
        return sb.toString();
    }

    static int [] randomTour(){
        int[] tour={0,1,2,3};
        Random rnd=new Random();

        for(int i=tour.length-1;i>0;i--){
            int j=rnd.nextInt(i+1);
            int temp=tour[i];

            tour[i]=tour[j];
            tour[j]=temp;

        }
        return tour;

    }

    static int [][] initPopulation(int popSize){
        int [][] population=new int[popSize][];
        for(int i=0;i<popSize;i++){
            population[i]=randomTour();
        }
        return population;
    }

    static void printPopulation(int [][] population){
        for(int i=0;i<population.length;i++){
            int[] path=population[i];
            System.out.println("个体 " + (i + 1) + ": " + formatTour(path)
                    + "  总距离=" + calcDistance(path));
        }
    }

    static int[] tournamentSelect(int[][] population,int tournamentSize){
        Random rnd=new Random();
        int bestIndex=-1;
        int bestDistance=Integer.MAX_VALUE;

        for(int i=0;i<tournamentSize;i++){
            int idx=rnd.nextInt(population.length);
            int[] candidate=population[idx];
            int dist=calcDistance(candidate);

            if(dist<bestDistance){
                bestDistance = dist;
                bestIndex = idx;
            }
        }
        int[] winner= Arrays.copyOf(population[bestIndex],population[bestIndex].length);
        return winner;
    }

    static int[] orderCrossover(int[] p1,int[] p2){
        int len=p1.length;
        int[] child=new int[len];

        Arrays.fill(child,-1);
        Random rnd=new Random();

        int i=rnd.nextInt(len);
        int j=rnd.nextInt(len);

        if(i>j){
            int tmp=i;
            i=j;
            j=tmp;
        }

        for(int k=i;k<=j;k++){
            child[k]=p1[k];
        }

        int cur=(j+1)%len;
        int p2idx=(j+1)%len;

        while(cur!=i){
            int city=p2[p2idx];

            boolean exists=false;

            for(int x=0;x<len;x++){
                if(child[x]==city){
                    exists = true;
                    break;
                }
            }

            if(!exists){
                child[cur]=city;
                cur = (cur + 1) % len;
            }
            p2idx=(p2idx + 1)%len;
        }
        return child;
    }

    static void swapMutation(int[] tour,double mutationRate){
        Random rnd=new Random();

        if(rnd.nextDouble()<mutationRate){
            int i = rnd.nextInt(tour.length);
            int j=rnd.nextInt(tour.length);
            while(i==j){
                j=rnd.nextInt(tour.length);
            }
            int temp = tour[i];
            tour[i] = tour[j];
            tour[j] = temp;
        }
    }

    static void runGA() {
        final int POP_SIZE = 8;
        final int GENERATIONS = 50;
        final double MUTATION_RATE = 0.10;

        int[][] population = initPopulation(POP_SIZE);

        int[] bestOverall = null;
        int bestOverallDist = Integer.MAX_VALUE;

        java.util.Random rnd = new java.util.Random();

        for (int gen = 1; gen <= GENERATIONS; gen++) {
            int[][] newPopulation = new int[POP_SIZE][];

            int bestIdx = 0;
            int bestDist = calcDistance(population[0]);
            for (int i = 1; i < POP_SIZE; i++) {
                int d = calcDistance(population[i]);
                if (d < bestDist) {
                    bestDist = d;
                    bestIdx = i;
                }
            }
            newPopulation[0] = java.util.Arrays.copyOf(population[bestIdx], population[bestIdx].length);

            if (bestDist < bestOverallDist) {
                bestOverallDist = bestDist;
                bestOverall = java.util.Arrays.copyOf(population[bestIdx], population[bestIdx].length);
            }

            for (int i = 1; i < POP_SIZE; i++) {
                int[] parent1 = tournamentSelect(population, 3);
                int[] parent2 = tournamentSelect(population, 3);

                int[] child = orderCrossover(parent1, parent2);

                swapMutation(child, MUTATION_RATE);

                newPopulation[i] = child;
            }

            population = newPopulation;

            System.out.printf("第 %2d 代最优路径: %-25s  距离=%d%n",
                    gen, formatTour(newPopulation[0]), bestDist);
        }
        System.out.println("\n=== 最终结果 ===");
        System.out.println("最短距离 = " + bestOverallDist);
        System.out.println("最优路径 = " + formatTour(bestOverall));
    }


    public static void main(String[] args) {
        System.out.println("=== 遗传算法求解 4 城市 TSP ===");
        runGA();
    }
}

