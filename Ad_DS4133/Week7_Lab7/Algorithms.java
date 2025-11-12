
// Algorithms.java
import java.util.*;

public class Algorithms {

    // =============== 1️⃣ BFS（广度优先遍历）================
    public static class BFS {
        public static <V> List<Vertex<V>> traverse(Graph<V> g, Vertex<V> start) {
            List<Vertex<V>> order = new ArrayList<>();
            Set<Vertex<V>> visited = new HashSet<>();
            Queue<Vertex<V>> q = new LinkedList<>();
            visited.add(start);
            q.add(start);
            while (!q.isEmpty()) {
                Vertex<V> v = q.poll();
                order.add(v);
                for (Edge<V> e : g.outgoingEdges(v)) {
                    Vertex<V> u = e.getEnd();
                    if (!visited.contains(u)) {
                        visited.add(u);
                        q.add(u);
                    }
                }
            }
            return order;
        }
    }

    // =============== 2️⃣ DFS（深度优先遍历）================
    public static class DFS {
        public static <V> List<Vertex<V>> traverse(Graph<V> g, Vertex<V> start) {
            List<Vertex<V>> order = new ArrayList<>();
            Set<Vertex<V>> visited = new HashSet<>();
            dfs(g, start, visited, order);
            return order;
        }

        private static <V> void dfs(Graph<V> g, Vertex<V> v, Set<Vertex<V>> visited, List<Vertex<V>> order) {
            visited.add(v);
            order.add(v);
            for (Edge<V> e : g.outgoingEdges(v)) {
                Vertex<V> u = e.getEnd();
                if (!visited.contains(u))
                    dfs(g, u, visited, order);
            }
        }
    }

    // =============== 3️⃣ Dijkstra（单源最短路径）================
    public static class Dijkstra {
        public static <V> Map<Vertex<V>, Double> shortestPath(Graph<V> g, Vertex<V> source) {
            Map<Vertex<V>, Double> dist = new HashMap<>();
            for (Vertex<V> v : g.vertices())
                dist.put(v, Double.POSITIVE_INFINITY);
            dist.put(source, 0.0);

            PriorityQueue<Vertex<V>> pq = new PriorityQueue<>(Comparator.comparingDouble(dist::get));
            pq.add(source);

            while (!pq.isEmpty()) {
                Vertex<V> u = pq.poll();
                double d = dist.get(u);
                for (Edge<V> e : g.outgoingEdges(u)) {
                    Vertex<V> v = e.getEnd();
                    double nd = d + e.getWeight();
                    if (nd < dist.get(v)) {
                        dist.put(v, nd);
                        pq.remove(v);
                        pq.add(v);
                    }
                }
            }
            return dist;
        }
    }

    // =============== 4️⃣ Topological Sort + Cycle Check =================
    public static class TopoSort {
        public static <V> List<Vertex<V>> sort(Graph<V> g) {
            Map<Vertex<V>, Integer> indeg = new HashMap<>();
            for (Vertex<V> v : g.vertices())
                indeg.put(v, g.inDegree(v));

            Queue<Vertex<V>> q = new LinkedList<>();
            for (Vertex<V> v : indeg.keySet())
                if (indeg.get(v) == 0)
                    q.add(v);

            List<Vertex<V>> order = new ArrayList<>();
            while (!q.isEmpty()) {
                Vertex<V> v = q.poll();
                order.add(v);
                for (Edge<V> e : g.outgoingEdges(v)) {
                    Vertex<V> u = e.getEnd();
                    indeg.put(u, indeg.get(u) - 1);
                    if (indeg.get(u) == 0)
                        q.add(u);
                }
            }

            if (order.size() != g.numVertices()) {
                System.out.println("⚠️ 图中存在环！");
            }
            return order;
        }
    }

    // =============== 5️⃣ Prim（最小生成树）================
    public static class Prim {
        public static <V> List<Edge<V>> mst(Graph<V> g) {
            if (g.isDirected())
                throw new IllegalArgumentException("Prim requires undirected graph");
            List<Edge<V>> mstEdges = new ArrayList<>();
            Set<Vertex<V>> inTree = new HashSet<>();

            Vertex<V> start = g.vertices().iterator().next();
            inTree.add(start);

            PriorityQueue<Edge<V>> pq = new PriorityQueue<>(Comparator.comparingInt(Edge::getWeight));
            for (Edge<V> edge : g.outgoingEdges(start)) {
                pq.add(edge);
            }

            while (!pq.isEmpty() && inTree.size() < g.numVertices()) {
                Edge<V> e = pq.poll();
                Vertex<V> v = e.getEnd();
                if (inTree.contains(v))
                    continue;
                inTree.add(v);
                mstEdges.add(e);
                for (Edge<V> ne : g.outgoingEdges(v)) {
                    if (!inTree.contains(ne.getEnd()))
                        pq.add(ne);
                }
            }
            return mstEdges;
        }
    }
}
