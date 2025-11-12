// AdjListGraph.java
import java.util.*;

public class AdjListGraph<V> implements Graph<V> {
    private final boolean directed;
    private final List<Vertex<V>> verts = new ArrayList<>();
    private final List<List<Edge<V>>> adj = new ArrayList<>();
    private int edgeCount = 0;

    public AdjListGraph(boolean directed){
        this.directed = directed;
    }

    @Override public boolean isDirected(){ return directed; }
    @Override public int numVertices(){ return verts.size(); }
    @Override public int numEdges(){ return edgeCount; }

    @Override public Iterable<Vertex<V>> vertices(){ return Collections.unmodifiableList(verts); }

    @Override
    public Iterable<Edge<V>> edges() {
        List<Edge<V>> all = new ArrayList<>();
        for (List<Edge<V>> list : adj) all.addAll(list);
        return all;
    }

    @Override
    public Edge<V> getEdge(Vertex<V> u, Vertex<V> v) {
        checkVertex(u); checkVertex(v);
        for (Edge<V> e : adj.get(u.id)) {
            if (e.getEnd().equals(v)) return e;
        }
        return null;
    }

    @Override
    public int outDegree(Vertex<V> v) {
        checkVertex(v);
        return adj.get(v.id).size();
    }

    @Override
    public int inDegree(Vertex<V> v) {
        checkVertex(v);
        int count=0;
        for (List<Edge<V>> list : adj)
            for (Edge<V> e : list)
                if (e.getEnd().equals(v)) count++;
        return count;
    }

    @Override
    public Iterable<Edge<V>> outgoingEdges(Vertex<V> v) {
        checkVertex(v);
        return Collections.unmodifiableList(adj.get(v.id));
    }

    @Override
    public Iterable<Edge<V>> incomingEdges(Vertex<V> v) {
        List<Edge<V>> res = new ArrayList<>();
        for (List<Edge<V>> list : adj)
            for (Edge<V> e : list)
                if (e.getEnd().equals(v)) res.add(e);
        return res;
    }

    @Override
    public Vertex<V> insertVertex(V x) {
        Vertex<V> nv = new Vertex<>(verts.size(), x);
        verts.add(nv);
        adj.add(new ArrayList<>());
        return nv;
    }

    @Override
    public Edge<V> insertEdge(Vertex<V> u, Vertex<V> v, int w) {
        checkVertex(u); checkVertex(v);
        Edge<V> e = new Edge<>(u, v, w);
        adj.get(u.id).add(e);
        if (!directed) adj.get(v.id).add(new Edge<>(v, u, w));
        edgeCount++;
        return e;
    }

    @Override
    public void removeEdge(Edge<V> e) {
        Vertex<V> u = e.getStart();
        Vertex<V> v = e.getEnd();
        checkVertex(u);
        checkVertex(v);
        boolean removed = adj.get(u.id).remove(e);
        if (!removed) return;
        if (!directed) {
            Iterator<Edge<V>> it = adj.get(v.id).iterator();
            while (it.hasNext()) {
                Edge<V> rev = it.next();
                if (rev.getEnd() == u && rev.getWeight() == e.getWeight()) {
                    it.remove();
                    break;
                }
            }
        }
        edgeCount--;
    }

    @Override
    public void removeVertex(Vertex<V> v) {
        checkVertex(v);
        int id = v.id;
        for (List<Edge<V>> list : adj) {
            list.removeIf(e -> e.getEnd() == v);
        }
        adj.remove(id);
        verts.remove(id);
        for (int i = id; i < verts.size(); i++) {
            Vertex<V> vertex = verts.get(i);
            vertex.setId(i);
        }
        edgeCount = recomputeEdgeCount();
    }

    private void checkVertex(Vertex<V> v){
        if (v==null || v.id<0 || v.id>=verts.size())
            throw new IllegalArgumentException("Invalid vertex");
    }

    private int recomputeEdgeCount() {
        int total = 0;
        for (List<Edge<V>> list : adj) total += list.size();
        return directed ? total : total / 2;
    }
}
