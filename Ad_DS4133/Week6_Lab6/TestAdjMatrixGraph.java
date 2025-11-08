public class TestAdjMatrixGraph {
    public static void main(String[] args) {
        AdjMatrixGraph<String, Integer> g = new AdjMatrixGraph<>(false); // 无向图

        Vertex<String>[] v = new Vertex[10];
        v[0] = g.insertVertex("Beijing");
        v[1] = g.insertVertex("Tianjin");
        v[2] = g.insertVertex("Chengde");
        v[3] = g.insertVertex("Shijiazhuang");
        v[4] = g.insertVertex("Hohhot");
        v[5] = g.insertVertex("Jinan");
        v[6] = g.insertVertex("Taiyuan");
        v[7] = g.insertVertex("Zhengzhou");
        v[8] = g.insertVertex("Shenyang");
        v[9] = g.insertVertex("Dalian");

        g.insertEdge(v[0], v[1], 125);
        g.insertEdge(v[0], v[2], 226);
        g.insertEdge(v[0], v[3], 294);
        g.insertEdge(v[0], v[4], 486);
        g.insertEdge(v[0], v[6], 714);

        g.insertEdge(v[1], v[2], 271);
        g.insertEdge(v[1], v[3], 343);
        g.insertEdge(v[1], v[5], 384);
        g.insertEdge(v[1], v[8], 640);
        g.insertEdge(v[1], v[9], 791);

        g.insertEdge(v[2], v[8], 578);

        g.insertEdge(v[3], v[5], 312);
        g.insertEdge(v[3], v[6], 225);
        g.insertEdge(v[3], v[7], 420);

        g.insertEdge(v[4], v[6], 438);

        g.insertEdge(v[6], v[7], 437);

        g.insertEdge(v[5], v[7], 459);

        g.insertEdge(v[8], v[9], 392);
        System.out.println("Adjacency Matrix (distance in km):");
        printMatrix(g);
    }

    private static void printMatrix(AdjMatrixGraph<String, Integer> g) {
        int n = g.numVertices();
        System.out.print("      ");
        for (int i = 0; i < n; i++) System.out.printf("%6d", i);
        System.out.println();

        for (int i = 0; i < n; i++) {
            System.out.printf("%6d", i);
            for (int j = 0; j < n; j++) {
                Edge<Integer, String> e = g.getEdge(
                        g.vertices().iterator().next(), g.vertices().iterator().next());
            }
        }
    }
}
