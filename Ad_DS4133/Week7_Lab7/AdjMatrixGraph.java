// AdjMatrixGraph.java
import java.util.*;

public class AdjMatrixGraph<V> implements Graph<V> {
    private final boolean directed;
    private final List<Vertex<V>> verts = new ArrayList<>();
    private Edge<V>[][] matrix;
    private int edgeCount = 0;

    @SuppressWarnings("unchecked")
    public AdjMatrixGraph(boolean directed, int initialCapacity){
        this.directed = directed;
        int n = Math.max(4, initialCapacity);
        this.matrix = (Edge<V>[][]) new Edge[n][n];
    }

    @Override public boolean isDirected(){ return directed; }

    @Override public int numVertices(){ return verts.size(); }
    @Override public int numEdges(){ return edgeCount; }

    @Override public Iterable<Vertex<V>> vertices(){ return Collections.unmodifiableList(verts); }

    @Override
    public Iterable<Edge<V>> edges() {
        List<Edge<V>> es = new ArrayList<>();
        for (int i=0;i<verts.size();i++){
            for (int j=0;j<verts.size();j++){
                if (matrix[i][j]!=null) es.add(matrix[i][j]);
            }
        }
        return es;
    }

    @Override
    public Edge<V> getEdge(Vertex<V> u, Vertex<V> v) {
        checkVertex(u); checkVertex(v);
        return matrix[u.id][v.id];
    }

    @Override
    public int outDegree(Vertex<V> v) {
        checkVertex(v);
        int d=0, i=v.id, n=verts.size();
        for (int j=0;j<n;j++) if (matrix[i][j]!=null) d++;
        return d;
    }

    @Override
    public int inDegree(Vertex<V> v) {
        checkVertex(v);
        int d=0, j=v.id, n=verts.size();
        for (int i=0;i<n;i++) if (matrix[i][j]!=null) d++;
        return d;
    }

    @Override
    public Iterable<Edge<V>> outgoingEdges(Vertex<V> v) {
        checkVertex(v);
        List<Edge<V>> es = new ArrayList<>();
        int i=v.id, n=verts.size();
        for (int j=0;j<n;j++) if (matrix[i][j]!=null) es.add(matrix[i][j]);
        return es;
    }

    @Override
    public Iterable<Edge<V>> incomingEdges(Vertex<V> v) {
        checkVertex(v);
        List<Edge<V>> es = new ArrayList<>();
        int j=v.id, n=verts.size();
        for (int i=0;i<n;i++) if (matrix[i][j]!=null) es.add(matrix[i][j]);
        return es;
    }

    @Override
    public Vertex<V> insertVertex(V x) {
        Vertex<V> nv = new Vertex<>(verts.size(), x);
        verts.add(nv);
        ensureCapacity(verts.size());
        return nv;
    }

    @Override
    public Edge<V> insertEdge(Vertex<V> u, Vertex<V> v, int w) {
        checkVertex(u); checkVertex(v);
        if (matrix[u.id][v.id]!=null) throw new IllegalArgumentException("Edge already exists");
        Edge<V> e = new Edge<>(u, v, w);
        matrix[u.id][v.id] = e;
        if (!directed) {
            matrix[v.id][u.id] = new Edge<>(v, u, w);
        }
        edgeCount++;
        return e;
    }

    @Override
    public void removeEdge(Edge<V> e) {
        Vertex<V> u=e.u, v=e.v;
        if (matrix[u.id][v.id]==null) return;
        matrix[u.id][v.id]=null;
        if (!directed) matrix[v.id][u.id]=null;
        edgeCount--;
    }

    @Override
    public void removeVertex(Vertex<V> v) {
        checkVertex(v);
        int idx = v.id;
        for (int j=0;j<verts.size();j++) if (matrix[idx][j]!=null){ matrix[idx][j]=null; edgeCount--; }
        for (int i=0;i<verts.size();i++) if (matrix[i][idx]!=null){ matrix[i][idx]=null; if (directed) edgeCount--; }
        int last = verts.size()-1;
        if (idx != last){
            Vertex<V> tail = verts.get(last);
            verts.set(idx, new Vertex<>(idx, tail.element));
            for (int j=0;j<verts.size();j++) matrix[idx][j]=matrix[last][j];
            for (int i=0;i<verts.size();i++) matrix[i][idx]=matrix[i][last];
            for (int i=0;i<verts.size();i++){
                for (int j=0;j<verts.size();j++){
                    Edge<V> ed = matrix[i][j];
                    if (ed!=null){
                        Vertex<V> nu = verts.get(i);
                        Vertex<V> nv = verts.get(j);
                        matrix[i][j] = new Edge<>(nu, nv, ed.weight);
                    }
                }
            }
        }
        verts.remove(last);
        for (int j=0;j<=last;j++) matrix[last][j]=null;
        for (int i=0;i<=last;i++) matrix[i][last]=null;
    }

    private void checkVertex(Vertex<V> v){
        if (v==null || v.id<0 || v.id>=verts.size())
            throw new IllegalArgumentException("Vertex out of range");
    }

    @SuppressWarnings("unchecked")
    private void ensureCapacity(int n){
        if (n <= matrix.length) return;
        int m = Math.max(matrix.length*2, n);
        Edge<V>[][] nm = (Edge<V>[][]) new Edge[m][m];
        for (int i=0;i<verts.size();i++)
            System.arraycopy(matrix[i], 0, nm[i], 0, verts.size());
        matrix = nm;
    }
}
