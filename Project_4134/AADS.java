import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

/**
 * @author Mingyuan LIU
 * @Email:scxml3@nottingham.edu.cn/lfuturemy@163.com
 * @since 2025-10-18
 * @describe: Project in Advanced Algorithms and Data Structures (COMP4134 UNNC) - Debug Version
 */

public class AADS {

    public static void main(String[] args) {
        try {
            // 1️⃣ 从标准输入读取完整 JSON
            String jsonString = readFromStdin();

            // 2️⃣ 解析为 Instance（包含 metadata / directions / viewpoints / samplePoints / collisionMatrix）
            Instance ins = SimpleJsonParser.parseInstance(jsonString);

            // 3️⃣ 打印摘要
            printSummary(ins);

// ====== Step 5. 调用贪心算法 ======
            System.out.println("\n===== Running GreedySolver =====");
            Solution sol = GreedySolver.solve(ins);

// ====== Step 6. 输出结果 ======
            System.out.println("\n===== Solution (JSON) =====");
            System.out.println(sol.toJsonFormatted());
//            sol.saveToFile("solution.json");


        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static String readFromStdin() throws IOException {
        StringBuilder inputBuilder = new StringBuilder();
        try (BufferedReader in = new BufferedReader(new InputStreamReader(System.in))) {
            String line;
            while ((line = in.readLine()) != null) {
                inputBuilder.append(line);
            }
        }
        return inputBuilder.toString();
    }

    // ----------- 打印摘要（保持原逻辑） -----------

    private static void printSummary(Instance ins) {
        System.out.println("===== Metadata =====");
        if (ins.metadata != null) {
            System.out.println(ins.metadata);
        } else {
            System.out.println("null");
        }

        System.out.println("\n===== Directions (ALL) =====");
        if (ins.directions != null && !ins.directions.isEmpty()) {
            for (Direction d : ins.directions) {
                System.out.println(d);
            }
        } else {
            System.out.println("(empty)");
        }

        System.out.println("\n===== Viewpoints (first 5) =====");
        printFirstN(ins.viewpoints, 5, vp -> {
            StringBuilder sb = new StringBuilder();
            sb.append("Viewpoint{id='").append(vp.id).append("', mandatory=").append(vp.mandatory)
                    .append(", xyz=(").append(vp.x).append(", ").append(vp.y).append(", ").append(vp.z).append(")");
            sb.append(", precisionByDirKeys=").append(vp.precisionByDir == null ? 0 : vp.precisionByDir.size());
            if (vp.precisionByDir != null && !vp.precisionByDir.isEmpty()) {
                int shown = 0;
                sb.append(", head=");
                for (Map.Entry<String, Double> e : vp.precisionByDir.entrySet()) {
                    sb.append(e.getKey()).append(":").append(e.getValue());
                    shown++;
                    if (shown >= 3) break;
                    sb.append(", ");
                }
            }
            sb.append("}");
            return sb.toString();
        });

        System.out.println("\n===== SamplePoints (first 5) =====");
        printFirstN(ins.samplePoints, 5, sp -> {
            StringBuilder sb = new StringBuilder();
            sb.append("SamplePoint{id='").append(sp.id).append("', xyz=(")
                    .append(sp.x).append(", ").append(sp.y).append(", ").append(sp.z).append("), pairs=");
            if (sp.coveringPairs == null) {
                sb.append("0");
            } else {
                int n = Math.min(3, sp.coveringPairs.size());
                sb.append(sp.coveringPairs.size()).append(", head=");
                for (int i = 0; i < n; i++) {
                    DirectionPair p = sp.coveringPairs.get(i);
                    sb.append("(").append(p.viewpointId).append(",").append(p.directionId).append(")");
                    if (i < n - 1) sb.append(", ");
                }
            }
            sb.append("}");
            return sb.toString();
        });

        System.out.println("\n===== Collision Matrix (5x5 preview) =====");
        printCollisionPreview(ins.collisionMatrix, 5);

        System.out.println("\n===== Start Viewpoint Id =====");
        System.out.println(ins.startViewpointId);
    }

    private interface ToLine<T> { String render(T t); }

    private static <T> void printFirstN(List<T> list, int n, ToLine<T> mapper) {
        if (list == null || list.isEmpty()) {
            System.out.println("(empty)");
            return;
        }
        int limit = Math.min(n, list.size());
        for (int i = 0; i < limit; i++) {
            System.out.println(mapper.render(list.get(i)));
        }
        if (list.size() > limit) {
            System.out.println("... (" + (list.size() - limit) + " more)");
        }
    }

    private static void printCollisionPreview(CollisionMatrix cm, int n) {
        if (cm == null || cm.matrix == null || cm.matrix.length == 0) {
            System.out.println("(empty)");
            return;
        }
        int size = Math.min(n, cm.matrix.length);
        for (int i = 0; i < size; i++) {
            StringBuilder row = new StringBuilder();
            for (int j = 0; j < size; j++) {
                row.append(cm.matrix[i][j]);
                if (j < size - 1) row.append(' ');
            }
            if (cm.matrix.length > size) row.append(" ...");
            System.out.println(row);
        }
        if (cm.matrix.length > size) {
            System.out.println("... (" + (cm.matrix.length - size) + " more rows)");
        }
    }
}


/**
 * 一个简单的、手写的JSON解析器，仅使用Java标准库。
 * 专门为处理本项目特定的JSON格式而设计。
 */
class SimpleJsonParser {
    private final String json;
    private int index = 0;

    public static Map<String, Object> parse(String jsonString) {
        SimpleJsonParser parser = new SimpleJsonParser(jsonString);
        return parser.parseObject();
    }

    private SimpleJsonParser(String json) {
        this.json = json;
    }

    private Map<String, Object> parseObject() {
        Map<String, Object> map = new HashMap<>();
        skipWhitespace();
        expect('{');
        index++; // skip '{'

        while (true) {
            skipWhitespace();
            if (peek('}')) {
                index++; // skip '}'
                break;
            }

            String key = parseString();
            skipWhitespace();
            expect(':'); index++;

            Object value = parseValue();
            map.put(key, value);

            skipWhitespace();
            if (peek(',')) {
                index++; // skip ','
            } else if (peek('}')) {
                // loop will close next iteration
            } else {
                skipWhitespace();
            }
        }
        return map;
    }

    private List<Object> parseArray() {
        List<Object> list = new ArrayList<>();
        skipWhitespace();
        expect('['); index++;

        while (true) {
            skipWhitespace();
            if (peek(']')) { index++; break; }
            Object value = parseValue();
            list.add(value);
            skipWhitespace();
            if (peek(',')) { index++; }
        }
        return list;
    }

    private Object parseValue() {
        skipWhitespace();
        char c = json.charAt(index);
        if (c == '"') return parseString();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == 't' || c == 'f') return parseBoolean();
        if (c == 'n') return parseNull();
        return parseNumber();
    }

    private String parseString() {
        skipWhitespace();
        expect('"'); index++;
        StringBuilder sb = new StringBuilder();
        while (json.charAt(index) != '"') {
            // 简化：不处理转义字符，假设输入干净
            sb.append(json.charAt(index));
            index++;
        }
        index++; // skip closing '"'
        return sb.toString();
    }

    private Object parseNumber() {
        skipWhitespace();
        StringBuilder sb = new StringBuilder();
        while (index < json.length()) {
            char ch = json.charAt(index);
            if (Character.isDigit(ch) || ch == '.' || ch == '-' || ch == '+'
                    || ch == 'e' || ch == 'E') {
                sb.append(ch);
                index++;
            } else break;
        }
        String numStr = sb.toString();
        try {
            if (numStr.contains(".") || numStr.contains("e") || numStr.contains("E")) {
                return Double.parseDouble(numStr);
            } else {
                long asLong = Long.parseLong(numStr);
                if (asLong <= Integer.MAX_VALUE && asLong >= Integer.MIN_VALUE) {
                    return (int) asLong;
                }
                return asLong;
            }
        } catch (NumberFormatException e) {
            throw new RuntimeException("Invalid number format: " + numStr);
        }
    }

    private Boolean parseBoolean() {
        skipWhitespace();
        if (json.startsWith("true", index)) { index += 4; return true; }
        if (json.startsWith("false", index)) { index += 5; return false; }
        throw new RuntimeException("Expected boolean");
    }

    private Object parseNull() {
        skipWhitespace();
        if (json.startsWith("null", index)) { index += 4; return null; }
        throw new RuntimeException("Expected null");
    }

    private void skipWhitespace() {
        while (index < json.length() && Character.isWhitespace(json.charAt(index))) index++;
    }

    private void expect(char ch) {
        if (index >= json.length() || json.charAt(index) != ch) {
            throw new RuntimeException("Expected '" + ch + "' at index " + index);
        }
    }

    private boolean peek(char ch) {
        return index < json.length() && json.charAt(index) == ch;
    }

    // -------------------- 业务解析：转为 Instance --------------------

    @SuppressWarnings("unchecked")
    public static Instance parseInstance(String jsonString) {
        Map<String, Object> data = parse(jsonString);

        // 1️⃣ viewpoints
        List<Viewpoint> viewpoints = parseViewpoints(asList(data.get("viewpoints")));

        // 2️⃣ samplePoints：兼容 "samplePoints"（驼峰）和 "sample_points"（下划线）
        Object spRaw = data.get("samplePoints");
        if (spRaw == null) spRaw = data.get("sample_points");
        List<SamplePoint> samplePoints = parseSamplePoints(asList(spRaw));

        // 3️⃣ collision_matrix: 兼容两种格式：[[...]] 或 {"matrix":[[...]]}
        CollisionMatrix collisionMatrix = parseCollisionMatrix(data.get("collision_matrix"));

        // 4️⃣ metadata + 起点推断
        Map<String, Object> mdRaw = asMap(data.get("metadata"));
        String startId = null;

        // Step 4.1：优先从 metadata 读取
        if (mdRaw != null && mdRaw.get("startViewpointId") instanceof String) {
            startId = (String) mdRaw.get("startViewpointId");
        }

        // Step 4.2：顶层存在 startViewpointId
        if (startId == null && data.get("startViewpointId") instanceof String) {
            startId = (String) data.get("startViewpointId");
        }

        // Step 4.3：若仍为空，则自动检测 mandatory
        if (startId == null) {
            List<String> mandatoryIds = new ArrayList<>();
            for (Viewpoint v : viewpoints) {
                if (v.mandatory) mandatoryIds.add(v.id);
            }

            if (mandatoryIds.size() == 1) {
                startId = mandatoryIds.get(0);
                System.out.println("[INFO] Detected mandatory start viewpoint: " + startId);
            } else if (mandatoryIds.size() > 1) {
                startId = mandatoryIds.get(0);
                System.err.println("[WARN] Multiple mandatory viewpoints found; using " + startId);
            } else if (!viewpoints.isEmpty()) {
                startId = viewpoints.get(0).id;
                System.err.println("[WARN] No mandatory viewpoint found; using " + startId);
            } else {
                System.err.println("[ERROR] No viewpoints available!");
            }
        }

        // 5️⃣ directions：若没有给出，则从 precisionByDir 的 key 汇总
        List<Direction> directions = parseDirections(asList(data.get("directions")), viewpoints);

        // 6️⃣ metadata：若没有给出则推断
        Metadata metadata = parseMetadata(mdRaw, viewpoints, samplePoints, startId);

        // ✅ 最终返回实例
        return new Instance(viewpoints, samplePoints, collisionMatrix, metadata, directions, startId);
    }


    // ---------- 通用取值/类型助手 ----------

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object o) {
        return (o instanceof Map) ? (Map<String, Object>) o : null;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asList(Object o) {
        return (o instanceof List) ? (List<Object>) o : null;
    }

    private static String getString(Map<String, Object> m, String... keys) {
        if (m == null) return null;
        for (String k : keys) {
            Object v = m.get(k);
            if (v instanceof String) return (String) v;
        }
        return null;
    }

    private static Boolean getBoolean(Map<String, Object> m, String... keys) {
        if (m == null) return null;
        for (String k : keys) {
            Object v = m.get(k);
            if (v instanceof Boolean) return (Boolean) v;
            if (v instanceof String) {
                String s = ((String) v).trim().toLowerCase();
                if ("true".equals(s)) return true;
                if ("false".equals(s)) return false;
            }
            if (v instanceof Number) {
                return ((Number) v).intValue() != 0;
            }
        }
        return null;
    }

    private static Double getDouble(Map<String, Object> m, String... keys) {
        if (m == null) return null;
        for (String k : keys) {
            Object v = m.get(k);
            if (v instanceof Number) return ((Number) v).doubleValue();
            if (v instanceof String) {
                try { return Double.parseDouble((String) v); } catch (Exception ignore) {}
            }
        }
        return null;
    }

    private static Integer getInt(Object o) {
        if (o instanceof Number) return ((Number) o).intValue();
        if (o instanceof String) {
            try { return Integer.parseInt((String) o); } catch (Exception ignore) {}
        }
        return null;
    }

    // ---------- 各块解析 ----------

    @SuppressWarnings("unchecked")
    private static List<Viewpoint> parseViewpoints(List<Object> viewpointsData) {
        List<Viewpoint> viewpoints = new ArrayList<>();
        if (viewpointsData == null) return viewpoints;

        for (Object obj : viewpointsData) {
            Map<String, Object> vpData = asMap(obj);
            if (vpData == null) continue;

            String id = getString(vpData, "id", "Id", "ID", "name");
            if (id == null) {
                System.err.println("[WARN] viewpoint missing id, skipped: keys=" + vpData.keySet());
                continue;
            }

            Boolean mandatoryBox = getBoolean(vpData, "is_mandatory", "isMandatory", "mandatory");
            boolean mandatory = mandatoryBox != null ? mandatoryBox : false;

            Double x = getDouble(vpData, "x", "X");
            Double y = getDouble(vpData, "y", "Y");
            Double z = getDouble(vpData, "z", "Z");

            if (x == null || y == null || z == null) {
                Map<String, Object> pos = asMap(vpData.get("pos"));
                if (pos == null) pos = asMap(vpData.get("position"));
                if (pos == null) pos = asMap(vpData.get("coordinates"));  // ✅ 新增这一行
                if (pos != null) {
                    if (x == null) x = getDouble(pos, "x", "X");
                    if (y == null) y = getDouble(pos, "y", "Y");
                    if (z == null) z = getDouble(pos, "z", "Z");
                }
            }


            if (x == null) x = 0.0;
            if (y == null) y = 0.0;
            if (z == null) z = 0.0;

            Map<String, Double> precisionByDir = new HashMap<>();
            Map<String, Object> p =
                    asMap(vpData.get("precisionByDir"));
            if (p == null) p = asMap(vpData.get("precisionByDirection"));
            if (p == null) p = asMap(vpData.get("precision"));
            if (p == null) p = asMap(vpData.get("precisions"));

            if (p != null) {
                for (Map.Entry<String, Object> e : p.entrySet()) {
                    Double val = null;
                    Object v = e.getValue();
                    if (v instanceof Number) val = ((Number) v).doubleValue();
                    else if (v instanceof String) {
                        try { val = Double.parseDouble((String) v); } catch (Exception ignore) {}
                    }
                    if (val != null) precisionByDir.put(e.getKey(), val);
                }
            } else {
                System.err.println("[WARN] viewpoint " + id + " missing precisionByDir-like map");
            }

            viewpoints.add(new Viewpoint(id, mandatory, x, y, z, precisionByDir));
        }
        return viewpoints;
    }

    @SuppressWarnings("unchecked")
    private static List<SamplePoint> parseSamplePoints(List<Object> samplePointsData) {
        List<SamplePoint> samplePoints = new ArrayList<>();
        if (samplePointsData == null) return samplePoints;

        for (Object obj : samplePointsData) {
            Map<String, Object> spData = asMap(obj);
            if (spData == null) continue;

            String id = getString(spData, "id", "Id", "ID", "name");
            if (id == null) continue;

            // 坐标字段兼容 coordinates / position
            Map<String, Object> coord = asMap(spData.get("coordinates"));
            if (coord == null) coord = asMap(spData.get("position"));
            Double x = getDouble(coord, "x", "X");
            Double y = getDouble(coord, "y", "Y");
            Double z = getDouble(coord, "z", "Z");
            if (x == null) x = 0.0;
            if (y == null) y = 0.0;
            if (z == null) z = 0.0;

            // 覆盖对：支持 [["v1","a2"],["v2","a3"]] 和 [{"viewpointId":...,"directionId":...}]
            List<DirectionPair> coveringPairs = new ArrayList<>();
            List<Object> pairsData = asList(spData.get("covering_pairs"));
            if (pairsData == null) pairsData = asList(spData.get("coveringPairs"));
            if (pairsData != null) {
                for (Object pObj : pairsData) {
                    if (pObj instanceof List list && list.size() == 2) {
                        coveringPairs.add(new DirectionPair(
                                String.valueOf(list.get(0)),
                                String.valueOf(list.get(1))
                        ));
                    } else if (pObj instanceof Map) {
                        Map<String, Object> pairData = asMap(pObj);
                        String viewpointId = getString(pairData, "viewpointId", "vpId", "viewpoint_id");
                        String directionId = getString(pairData, "directionId", "dirId", "direction_id");
                        if (viewpointId != null && directionId != null)
                            coveringPairs.add(new DirectionPair(viewpointId, directionId));
                    }
                }
            }
            samplePoints.add(new SamplePoint(id, x, y, z, coveringPairs));
        }
        return samplePoints;
    }



    @SuppressWarnings("unchecked")
    private static CollisionMatrix parseCollisionMatrix(Object raw) {
        if (raw == null) return new CollisionMatrix(0);
        if (raw instanceof Map) {
            Map<String, Object> m = (Map<String, Object>) raw;
            Object maybe = m.get("matrix");
            if (maybe instanceof List) {
                return from2DList((List<Object>) maybe);
            }
        } else if (raw instanceof List) {
            return from2DList((List<Object>) raw);
        }
        throw new RuntimeException("collision_matrix format not recognized");
    }

    @SuppressWarnings("unchecked")
    private static CollisionMatrix from2DList(List<Object> matrixList) {
        int size = matrixList.size();
        CollisionMatrix cm = new CollisionMatrix(size);
        for (int i = 0; i < size; i++) {
            List<Object> row = (List<Object>) matrixList.get(i);
            for (int j = 0; j < row.size(); j++) {
                int val = ((Number) row.get(j)).intValue();
                cm.setCollision(i, j, val);
            }
        }
        return cm;
    }

    @SuppressWarnings("unchecked")
    private static List<Direction> parseDirections(List<Object> directionsData, List<Viewpoint> vps) {
        List<Direction> list = new ArrayList<>();
        if (directionsData == null || directionsData.isEmpty()) {
            // 没提供 directions，则自动从 viewpoints 汇总
            Set<String> keys = new LinkedHashSet<>();
            for (Viewpoint vp : vps) {
                if (vp.precisionByDir != null) keys.addAll(vp.precisionByDir.keySet());
            }
            for (String k : keys) list.add(new Direction(k, null));
            return list;
        }

        // 如果 directions 是 [[x,y,z], [x,y,z], ...] 格式
        if (directionsData.get(0) instanceof List) {
            int idx = 1;
            for (Object o : directionsData) {
                List<Object> coords = (List<Object>) o;
                if (coords.size() == 3) {
                    double dx = ((Number) coords.get(0)).doubleValue();
                    double dy = ((Number) coords.get(1)).doubleValue();
                    double dz = ((Number) coords.get(2)).doubleValue();
                    list.add(new Direction("a" + idx, dx, dy, dz));
                    idx++;
                }
            }
            return list;
        }

        // 如果 directions 是 [{"directionId": "...", "precision": ...}] 格式
        for (Object o : directionsData) {
            Map<String, Object> d = asMap(o);
            if (d == null) continue;
            String directionId = getString(d, "directionId", "dir", "id");
            Double precision = getDouble(d, "precision");
            list.add(new Direction(directionId, precision));
        }
        return list;
    }


    private static Metadata parseMetadata(Map<String, Object> mdRaw,
                                          List<Viewpoint> vps,
                                          List<SamplePoint> sps,
                                          String startId) {
        if (mdRaw != null) {
            Integer nv = getInt(mdRaw.get("numViewpoints"));
            Integer ns = getInt(mdRaw.get("numSamplePoints"));
            String sid = (String) mdRaw.getOrDefault("startViewpointId", startId);
            return new Metadata(
                    nv != null ? nv : (vps == null ? 0 : vps.size()),
                    ns != null ? ns : (sps == null ? 0 : sps.size()),
                    sid
            );
        }
        return new Metadata(vps == null ? 0 : vps.size(),
                sps == null ? 0 : sps.size(),
                startId);
    }
}

// -------------------- 业务数据结构 --------------------

class Viewpoint {
    String id;                // 视角ID
    boolean mandatory;        // 是否为强制视角（起点之一）
    double x, y, z;           // 坐标
    Map<String, Double> precisionByDir; // 方向 -> 精度

    public Viewpoint(String id, boolean mandatory, double x, double y, double z, Map<String, Double> precisionByDir) {
        this.id = id;
        this.mandatory = mandatory;
        this.x = x;
        this.y = y;
        this.z = z;
        this.precisionByDir = precisionByDir;
    }
}

class SamplePoint {
    String id;
    double x, y, z;
    List<DirectionPair> coveringPairs; // (viewpointId, directionId)；directionId 可能为 "all"

    public SamplePoint(String id, double x, double y, double z, List<DirectionPair> coveringPairs) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.z = z;
        this.coveringPairs = coveringPairs;
    }
}

class DirectionPair {
    String viewpointId;
    String directionId;

    public DirectionPair(String viewpointId, String directionId) {
        this.viewpointId = viewpointId;
        this.directionId = directionId;
    }
}

class CollisionMatrix {
    int[][] matrix;

    public CollisionMatrix(int size) {
        this.matrix = new int[size][size];
    }

    public void setCollision(int i, int j, int value) {
        matrix[i][j] = value;
        matrix[j][i] = value; // 保持对称
    }

    public int getCollision(int i, int j) {
        return matrix[i][j];
    }
}

class Direction {
    String directionId;
    Double precision; // 可空
    Double dx, dy, dz; // 三维方向向量，可空

    public Direction(String directionId, Double precision) {
        this.directionId = directionId;
        this.precision = precision;
    }

    public Direction(String directionId, double dx, double dy, double dz) {
        this.directionId = directionId;
        this.dx = dx;
        this.dy = dy;
        this.dz = dz;
    }

    @Override
    public String toString() {
        if (dx != null && dy != null && dz != null)
            return "Direction{" + directionId + " = [" + dx + ", " + dy + ", " + dz + "]}";
        if (precision != null)
            return "Direction{" + directionId + ", precision=" + precision + "}";
        return "Direction{" + directionId + "}";
    }
}


class Metadata {
    int numViewpoints;
    int numSamplePoints;
    String startViewpointId;

    public Metadata(int numViewpoints, int numSamplePoints, String startViewpointId) {
        this.numViewpoints = numViewpoints;
        this.numSamplePoints = numSamplePoints;
        this.startViewpointId = startViewpointId;
    }

    @Override
    public String toString() {
        return "Metadata{numViewpoints=" + numViewpoints +
                ", numSamplePoints=" + numSamplePoints +
                ", startViewpointId='" + startViewpointId + '\'' +
                '}';
    }
}

class Instance {
    List<Viewpoint> viewpoints;
    List<SamplePoint> samplePoints;
    CollisionMatrix collisionMatrix;
    Metadata metadata;
    List<Direction> directions;
    String startViewpointId;

    public Instance(List<Viewpoint> viewpoints,
                    List<SamplePoint> samplePoints,
                    CollisionMatrix collisionMatrix,
                    Metadata metadata,
                    List<Direction> directions,
                    String startViewpointId) {
        this.viewpoints = viewpoints;
        this.samplePoints = samplePoints;
        this.collisionMatrix = collisionMatrix;
        this.metadata = metadata;
        this.directions = directions;
        this.startViewpointId = startViewpointId;
    }

    /**
     * ✅ 调试增强版：计算 viewpoint 间欧式距离矩阵，并打印可达信息
     * - 若 i==j，则距离=0；
     * - 若 collisionMatrix.matrix[i][j] == -1，则视为不可达；
     * - 否则计算欧式距离；
     * - 最后打印每个 viewpoint 可达节点数量。
     */
    public double[][] computeDistanceMatrix() {
        int n = viewpoints.size();
        double[][] dist = new double[n][n];

        System.out.println("\n===== Computing Distance Matrix =====");
        int totalBlocked = 0;

        for (int i = 0; i < n; i++) {
            int reachableCount = 0;
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    dist[i][j] = 0.0;
                } else if (collisionMatrix != null
                        && collisionMatrix.matrix != null
                        && i < collisionMatrix.matrix.length
                        && j < collisionMatrix.matrix[i].length
                        && collisionMatrix.matrix[i][j] == -1) {
                    dist[i][j] = Double.POSITIVE_INFINITY;  // 不可达
                    totalBlocked++;
                } else {
                    Viewpoint a = viewpoints.get(i);
                    Viewpoint b = viewpoints.get(j);
                    double dx = a.x - b.x;
                    double dy = a.y - b.y;
                    double dz = a.z - b.z;
                    dist[i][j] = Math.sqrt(dx * dx + dy * dy + dz * dz);
                    reachableCount++;
                }
            }

            // 每个 viewpoint 的可达数量打印
            System.out.printf("[DEBUG] Viewpoint %-6s → reachable: %3d / %3d%n",
                    viewpoints.get(i).id, reachableCount, n - 1);
        }

        System.out.println("[INFO] Distance matrix computed: size = " + n + "x" + n);
        System.out.println("[INFO] Total blocked pairs (collision = -1): " + totalBlocked);

        // 可选：如果大多数点都不可达，提示警告
        double ratio = totalBlocked / (double) (n * n);
        if (ratio > 0.9) {
            System.err.println("[WARN] Over 90% of viewpoint pairs are blocked by collision matrix!");
            System.err.println("[HINT] Check if your collision_matrix in JSON was parsed correctly.");
        }

        return dist;
    }

    public int getViewpintIndex(String id){
        for (int i = 0; i < viewpoints.size(); i++) {
            if(viewpoints.get(i).id.equals(id)){
                return i;
            }
        }
        return -1;
    }
}


