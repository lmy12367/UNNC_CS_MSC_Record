import ADT.AbstractSortedMap;
import ADT.Entry;
import ADT.ArrayList;
import java.util.Comparator;

public class SearchTableMap<K,V> extends AbstractSortedMap<K,V> {
    private ArrayList<MapEntry<K,V>> table=new ArrayList<>();
    public SearchTableMap(){
        super();
    }
    public SearchTableMap(Comparator<K> comp){
        super(comp);
    }

    public int findIndex(K key,int low,int high){
        if(high<low){
            return high+1;
        }
        int mid=(low+high)/2;
        int comp=compare(key,table.get(mid));
        if(comp==0){
            return 0;
        }
        else if(comp<0){
            return findIndex(key,low,mid-1);
        }
        else{
            return findIndex(key,mid+1,high);
        }
    }

    public int findIndex(K key){
        return findIndex(key,0,table.size()-1);
    }


    @Override
    public Entry<K, V> firstEntry() {
        return null;
    }

    @Override
    public Entry<K, V> lastEntry() {
        return null;
    }

    @Override
    public Entry<K, V> ceilingEntry(K key) throws IllegalArgumentException {
        return null;
    }

    @Override
    public Entry<K, V> floorEntry(K key) throws IllegalArgumentException {
        return null;
    }

    @Override
    public Entry<K, V> lowerEntry(K key) throws IllegalArgumentException {
        return null;
    }

    @Override
    public Entry<K, V> higherEntry(K key) throws IllegalArgumentException {
        return null;
    }

    @Override
    public Iterable<Entry<K, V>> subMap(K fromKey, K toKey) throws IllegalArgumentException {
        return null;
    }

    @Override
    public int size() {
        return table.size();
    }

    @Override
    public V get(K key) {
        int j=findIndex(key);
        if(j==size()||compare(key,table.get(j))!=0){
            return null;
        }
        return table.get(j).getValue();

    }
    @Override
    public V put(K key,V value){
        int j=findIndex(key);
        if(j<size()&&compare(key, table.get(j))==0){
            return table.get(j).setValue(value);
        }
        table.add(j,new MapEntry<K,V>(key,value));
        return null;
    }

    @Override
    public V remove(K key) {
        int j=findIndex(key);
        if(j==size()||compare(key,table.get(j))!=0){
            return null;
        }
        return table.remove(j).getValue();
    }

    @Override
    public Iterable<Entry<K, V>> entrySet() {
        return null;
    }
}
