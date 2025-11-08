class Vertex<V>{
    private final V  element;
    private final int id;

    public Vertex(V element, int id){
        this.element = element;
        this.id = id;
    }

    public V getElement(){
        return element;
    }

    public int getId(){
        return id;
    }

    @Override
    public String toString() {
        return "Vertex(" + element + ")" + "ID(" + id + ")";
    }
}

class Edge<E,V>{
    private final Vertex<V> u;
    private final Vertex<V> v;
    private final E element;

    public Edge(Vertex<V> u, Vertex<V> v, E element) {
        this.u = u;
        this.v = v;
        this.element = element;
    }

    public Vertex<V> getU(){
        return u;
    }

    public Vertex<V> getV(){
        return v;
    }

    public E getElement(){
        return element;
    }

    @Override
    public String toString(){
        return "E(" + u.getElement() + " -> " + v.getElement() + ", w=" + element + ")";
    }
}

public interface Graph <V,E> {
    int numVertices();
    int numEdges();

    Iterable<Vertex<V>> vertices();
    Iterable<Edge<E,V>> edges();

    Vertex<V> insertVertex(V x);
    Edge<E,V> insertEdge(Vertex<V> u,Vertex<V> v,E x);

    Edge<E,V> getEdge(Vertex<V> u,Vertex<V> v);

}
