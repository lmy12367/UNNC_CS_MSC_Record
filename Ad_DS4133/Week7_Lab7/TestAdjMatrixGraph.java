// TestAdjMatrixGraph.java
import java.util.*;

public class TestAdjMatrixGraph {
    public static void main(String[] args) {
        AdjMatrixGraph<String> g = new AdjMatrixGraph<>(false, 8); // 无向图
        Vertex<String> A = g.insertVertex("A");
        Vertex<String> B = g.insertVertex("B");
        Vertex<String> C = g.insertVertex("C");
        Vertex<String> D = g.insertVertex("D");
        Vertex<String> E = g.insertVertex("E");
        Vertex<String> F = g.insertVertex("F");

        g.insertEdge(A,B,1);
        g.insertEdge(A,D,1);
        g.insertEdge(B,E,1);
        g.insertEdge(C,F,1);
        g.insertEdge(D,E,1);
        g.insertEdge(E,F,1);

        System.out.println("V=" + g.numVertices() + ", E=" + g.numEdges());
        System.out.println("deg(A): out=" + g.outDegree(A) + ", in=" + g.inDegree(A));
        System.out.println("Edges:");
        for (Edge<String> e : g.edges()) System.out.println("  " + e);


        Edge<String> ab = g.getEdge(A,B);
        g.removeEdge(ab);
        System.out.println("After removing A-B, E=" + g.numEdges());

        g.removeVertex(C);
        System.out.println("After removing C, V=" + g.numVertices() + ", E=" + g.numEdges());
    }
}
