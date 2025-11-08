import java.util.ArrayList;
import java.util.List;

public class AdjMatrixGraph<V, E> implements Graph<V, E> {
    private final List<Vertex<V>> verts = new ArrayList<>();
    private Edge<E, V>[][] matrix;
    private int edgeCount = 0;
    private boolean directed;

    @SuppressWarnings("unchecked")
    public AdjMatrixGraph(boolean directed) {
        this.directed = directed;
        this.matrix = (Edge<E, V>[][]) new Edge[0][0];
    }

    @Override
    public int numVertices() {
        return verts.size();
    }

    @Override
    public int numEdges() {
        return edgeCount;
    }

    @Override
    public Iterable<Vertex<V>> vertices() {
        return verts;
    }

    @Override
    public Iterable<Edge<E, V>> edges() {
        List<Edge<E, V>> list = new ArrayList<>();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] != null) {
                    list.add(matrix[i][j]); // ✅ 修复 j->j 错误
                }
            }
        }
        return list;
    }

    @Override
    public Vertex<V> insertVertex(V x) {
        Vertex<V> v = new Vertex<>(x, verts.size());
        verts.add(v);
        growIfNeeded();
        return v;
    }

    @Override
    public Edge<E, V> insertEdge(Vertex<V> u, Vertex<V> v, E x) {
        int i = u.getId(), j = v.getId();
        if (matrix[i][j] != null) throw new IllegalArgumentException("Edge already exists");
        Edge<E, V> e = new Edge<>(u, v, x);
        matrix[i][j] = e;
        if (!directed) matrix[j][i] = new Edge<>(v, u, x);
        edgeCount++;
        return e;
    }

    @Override
    public Edge<E, V> getEdge(Vertex<V> u, Vertex<V> v) {
        return matrix[u.getId()][v.getId()];
    }

    @SuppressWarnings("unchecked")
    private void growIfNeeded() {
        int n = verts.size();
        if (matrix.length >= n) return;
        Edge<E, V>[][] newM = (Edge<E, V>[][]) new Edge[n][n];
        for (int i = 0; i < matrix.length; i++) {
            System.arraycopy(matrix[i], 0, newM[i], 0, matrix.length);
        }
        matrix = newM;
    }

    public int outDegree(Vertex<V> v) {
        int id = v.getId();
        int count = 0;
        for (int j = 0; j < matrix[id].length; j++) {
            if (matrix[id][j] != null) count++;
        }
        return count;
    }

    public int inDegree(Vertex<V> v) {
        int id = v.getId();
        int count = 0;
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][id] != null) count++;
        }
        return count;
    }

    public Iterable<Edge<E, V>> outgoingEdges(Vertex<V> v) {
        List<Edge<E, V>> list = new ArrayList<>();
        int id = v.getId();
        for (int j = 0; j < matrix.length; j++) {
            if (matrix[id][j] != null) list.add(matrix[id][j]);
        }
        return list;
    }

    public Iterable<Edge<E, V>> incomingEdges(Vertex<V> v) {
        List<Edge<E, V>> list = new ArrayList<>();
        int id = v.getId();
        for (int i = 0; i < matrix.length; i++) {
            if (matrix[i][id] != null) list.add(matrix[i][id]);
        }
        return list;
    }

    public void removeEdge(Vertex<V> u, Vertex<V> v) {
        int i = u.getId(), j = v.getId();
        if (matrix[i][j] != null) {
            matrix[i][j] = null;
            if (!directed) matrix[j][i] = null;
            edgeCount--;
        }
    }

    @SuppressWarnings("unchecked")
    public void removeVertex(Vertex<V> v) {
        int n = verts.size();
        int id = v.getId();

        for (int i = 0; i < n; i++) {
            if (matrix[id][i] != null) edgeCount--;
            if (matrix[i][id] != null) edgeCount--;
        }

        Edge<E, V>[][] newM = (Edge<E, V>[][]) new Edge[n - 1][n - 1];
        for (int i = 0, ni = 0; i < n; i++) {
            if (i == id) continue;
            for (int j = 0, nj = 0; j < n; j++) {
                if (j == id) continue;
                newM[ni][nj] = matrix[i][j];
                nj++;
            }
            ni++;
        }
        matrix = newM;

        verts.remove(id);
        for (int k = id; k < verts.size(); k++) {
            Vertex<V> old = verts.get(k);
            verts.set(k, new Vertex<>(old.getElement(), k));
        }

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                if (matrix[i][j] != null) {
                    Vertex<V> u = verts.get(i);
                    Vertex<V> w = verts.get(j);
                    matrix[i][j] = new Edge<>(u, w, matrix[i][j].getElement());
                }
            }
        }
    }
}
