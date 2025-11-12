// TestAdjListGraph.java
public class TestAdjListGraph {
    public static void main(String[] args) {
        AdjListGraph<String> g = new AdjListGraph<>(false);
        Vertex<String> A = g.insertVertex("A");
        Vertex<String> B = g.insertVertex("B");
        Vertex<String> C = g.insertVertex("C");
        g.insertEdge(A, B, 1);
        g.insertEdge(B, C, 1);
        g.insertEdge(A, C, 2);

        System.out.println("Vertices: " + g.numVertices());
        System.out.println("Edges: " + g.numEdges());
        System.out.println("Outgoing A: " + g.outDegree(A));
        System.out.println("All edges:");
        for (Edge<String> e : g.edges()) System.out.println("  " + e);
    }
}
