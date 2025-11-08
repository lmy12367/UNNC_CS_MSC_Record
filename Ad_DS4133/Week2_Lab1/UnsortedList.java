package ExOne;
import java.util.*;


public class UnsortedList<K,V> extends AbstractMap<K,V> {

    public static void main(String[] args) {
        Map<String, Integer> roster = new UnsortedList<>();
        roster.put("Alice", 90);
        roster.put("Bob", 80);
        roster.put("Alice", 95);
        System.out.println("Alice 小红花: " + roster.get("Alice"));
        roster.remove("Bob");
        System.out.println("现在人数: " + roster.size());
        System.out.println("所有名字: " + roster.keySet());
        System.out.println("所有花朵: " + roster.values());

    }

    private ArrayList<MapEntry<K, V>> table = new ArrayList<>();

    @Override
    public int size() {
        return table.size();
    }

    @Override
    public V get(Object key) {
        for (MapEntry<K, V> kvMapEntry : table) {
            if (kvMapEntry.getKey().equals(key)) {
                return kvMapEntry.getValue();
            }
        }
        return null;
    }

    @Override
    public V put(K key, V value) {
        for (MapEntry<K, V> kvMapEntry : table) {
            if (kvMapEntry.getValue().equals(key)) {
                V old = kvMapEntry.getValue();
                kvMapEntry.setValue(value);
                return old;
            }
        }
        table.add(new MapEntry<>(key, value));
        return null;
    }

    @Override
    public V remove(Object key) {
        Iterator<MapEntry<K, V>> it = table.iterator();
        while (it.hasNext()) {
            MapEntry<K, V> kvMapEntry = it.next();
            if (kvMapEntry.getKey().equals(key)) {
                V old = kvMapEntry.getValue();
                it.remove();
                return old;
            }
        }
        return null;
    }
    @Override
    public Set<Map.Entry<K, V>> entrySet() {
        return new AbstractSet<Map.Entry<K, V>>() {
            public Iterator<Map.Entry<K, V>> iterator() {
                return new Iterator<Map.Entry<K, V>>() {
                    private Iterator<MapEntry<K, V>> it = table.iterator();
                    public boolean hasNext() { return it.hasNext(); }
                    public Map.Entry<K, V> next() { return it.next(); }
                    public void remove() { it.remove(); }
                };
            }
            public int size() { return table.size(); }
        };
    }

    @Override
    public boolean containsKey(Object key) {
        return get(key) != null;
    }

    @Override
    public boolean containsValue(Object value) {
        for (MapEntry<K, V> entry : table) {
            if (entry.getValue().equals(value)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
        for (Map.Entry<? extends K, ? extends V> e : m.entrySet()) {
            put(e.getKey(), e.getValue());
        }
    }

    @Override
    public void clear() {
        table.clear();
    }

}



