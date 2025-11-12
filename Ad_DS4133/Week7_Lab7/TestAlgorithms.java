// TestAlgorithms.java
import java.util.*;

public class TestAlgorithms {
    public static void main(String[] args) {
        AdjMatrixGraph<String> g = new AdjMatrixGraph<>(false, 10);
        Vertex<String> A = g.insertVertex("A");
        Vertex<String> B = g.insertVertex("B");
        Vertex<String> C = g.insertVertex("C");
        Vertex<String> D = g.insertVertex("D");
        Vertex<String> E = g.insertVertex("E");
        Vertex<String> F = g.insertVertex("F");

        g.insertEdge(A, B, 2);
        g.insertEdge(A, C, 4);
        g.insertEdge(B, C, 1);
        g.insertEdge(B, D, 7);
        g.insertEdge(C, E, 3);
        g.insertEdge(D, F, 1);
        g.insertEdge(E, F, 5);

        System.out.println("========== BFS ==========");
        System.out.println(Algorithms.BFS.traverse(g, A));

        System.out.println("========== DFS ==========");
        System.out.println(Algorithms.DFS.traverse(g, A));

        System.out.println("========== Dijkstra ==========");
        System.out.println(Algorithms.Dijkstra.shortestPath(g, A));

        System.out.println("========== TopoSort (for DAG only) ==========");
        AdjMatrixGraph<String> dag = new AdjMatrixGraph<>(true, 6);
        Vertex<String> a = dag.insertVertex("a");
        Vertex<String> b = dag.insertVertex("b");
        Vertex<String> c = dag.insertVertex("c");
        Vertex<String> d = dag.insertVertex("d");
        dag.insertEdge(a,b,1);
        dag.insertEdge(a,c,1);
        dag.insertEdge(b,d,1);
        dag.insertEdge(c,d,1);
        System.out.println(Algorithms.TopoSort.sort(dag));

        System.out.println("========== Prim (MST) ==========");
        for (Edge<String> e : Algorithms.Prim.mst(g))
            System.out.println(e);
    }
}
