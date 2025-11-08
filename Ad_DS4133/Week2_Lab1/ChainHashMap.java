package ExOne;
import java.util.*;


public class ChainHashMap<K,V> extends AbstractHashMap<K,V> {
    /* ---------------- 桶数组 ---------------- */
    private LinkedList<Entry<K,V>>[] table;

    /* ---------------- 构造器 ---------------- */
    public ChainHashMap() { super(); }
    public ChainHashMap(int cap) { super(cap); }

    /* 1. 创建空表 */
    @Override
    @SuppressWarnings("unchecked")
    protected void createTable() {
        table = new LinkedList[capacity];
        for (int i = 0; i < capacity; i++) table[i] = new LinkedList<>();
    }

    /* 2. 桶内查找 */
    @Override
    protected V bucketGet(int h, K k) {
        for (Entry<K,V> e : table[h])
            if (e.getKey().equals(k)) return e.getValue();
        return null;
    }

    /* 3. 桶内插入/更新 */
    @Override
    protected V bucketPut(int h, K k, V v) {
        for (Entry<K,V> e : table[h]) {
            if (e.getKey().equals(k)) {          // 更新旧值
                V old = e.getValue();
                e.setValue(v);
                return old;
            }
        }
        table[h].addLast(new MapEntry<>(k, v));  // 新节点
        n++;                                     // 总条目 +1
        return null;
    }

    /* 4. 桶内删除 */
    @Override
    protected V bucketRemove(int h, K k) {
        Iterator<Entry<K,V>> it = table[h].iterator();
        while (it.hasNext()) {
            Entry<K,V> e = it.next();
            if (e.getKey().equals(k)) {
                V ans = e.getValue();
                it.remove();
                n--;
                return ans;
            }
        }
        return null;
    }

    /* 5. 返回 Set<Entry<K,V>> 供父类使用 */
    @Override
    public Set<Entry<K,V>> entrySet() {
        Set<Entry<K,V>> set = new HashSet<>();
        for (LinkedList<Entry<K,V>> chain : table)
            set.addAll(chain);
        return set;
    }

    /* 6. 可选但推荐：快速 containsKey */
    @Override
    public boolean containsKey(Object key) {
        return get(key) != null;
    }

    @Override
    public boolean containsValue(Object value) {
        for (Entry<K,V> e : entrySet())
            if (Objects.equals(value, e.getValue())) return true;
        return false;
    }

    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
        for (Entry<? extends K, ? extends V> e : m.entrySet()) {
            put(e.getKey(), e.getValue());
        }
    }
    @Override
    public void clear() {
        n = 0;                     // 条目数归零
        for (var chain : table)    // 清空所有链表
            chain.clear();
    }

    /* ---------------- 简单测试 ---------------- */
    public static void main(String[] args) {
        Map<String,Integer> map = new ChainHashMap<>();
        map.put("A", 1);
        map.put("B", 2);
        map.put("C", 3);
        System.out.println("get B -> " + map.get("B"));      // 2
        System.out.println("contains D? " + map.containsKey("D")); // false
        System.out.println("remove A -> " + map.remove("A")); // 1
        System.out.println("size after remove -> " + map.size()); // 2
        System.out.println("entrySet -> " + map.entrySet());
    }
}