public interface Graph<V> {
    int numVertices();
    int numEdges();

    Iterable<Vertex<V>> vertices();
    Iterable<Edge<V>> edges();

    Edge<V> getEdge(Vertex<V> u, Vertex<V> v);

    int outDegree(Vertex<V> v);
    int inDegree(Vertex<V> v);

    Iterable<Edge<V>> outgoingEdges(Vertex<V> v);
    Iterable<Edge<V>> incomingEdges(Vertex<V> v);

    Vertex<V> insertVertex(V x);
    Edge<V> insertEdge(Vertex<V> u, Vertex<V> v, int w); // w 为权重

    void removeEdge(Edge<V> e);
    void removeVertex(Vertex<V> v);

    boolean isDirected();
}

class Vertex<V> {
    int id;
    final V element;
    Vertex(int id, V element) { this.id = id; this.element = element; }
    public V getElement() { return element; }
    void setId(int id) { this.id = id; }
    @Override public String toString(){ return String.valueOf(element); }
}

class Edge<V> {
    final Vertex<V> u, v;
    final int weight;
    Edge(Vertex<V> u, Vertex<V> v, int w){ this.u=u; this.v=v; this.weight=w; }
    public Vertex<V> getStart(){ return u; }
    public Vertex<V> getEnd(){ return v; }
    public int getWeight(){ return weight; }
    @Override public String toString(){ return u+"->"+v+"(w="+weight+")"; }
}
