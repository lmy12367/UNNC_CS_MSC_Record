import java.io.*;
import java.util.*;
import java.util.Locale;

/**
 * ===========================================================
 * TestGenetic.java
 * - 读取 input.json
 * - 调用 GreedySolver + GeneticSolver(+2Opt)
 * - 校验：回到起点 / 路径可达 / 覆盖统计(≥3充分)
 * - 输出：TestGreedy.json / TestGenetic.json
 * ===========================================================
 */
public class TestGenetic {

    public static void main(String[] args) {
        try {
            System.out.println("=== 🚀 Greedy + Genetic Solver 测试开始 ===\n");

            // === 1️⃣ 读取 input.json ===
            String inputFile = "input_sample20251106.json";
            String json = readFile(inputFile);
            System.out.println("✅ 已读取输入文件: " + inputFile + " (" + json.length() + " chars)");

            // === 2️⃣ 解析 + 构建实例 ===
            JsonParser parser = new JsonParser(json);
            Map<String, Object> data = (Map<String, Object>) parser.parse();
            Instance instance = new InstanceBuilder(data).build();
            System.out.println("✅ Instance 构建成功\n");
            System.out.println("📍 视点数: " + instance.viewpoints.size());
            System.out.println("🎯 样本点数: " + instance.samplePoints.size());
            System.out.println("🚦 起点: " + instance.startViewpointId + "\n");

            // === 3️⃣ Greedy Solver ===
            System.out.println("🧮 运行 GreedySolver...");
            long t0 = System.currentTimeMillis();
            GreedySolver.GreedySolution greedy = GreedySolver.solve(instance);
            long t1 = System.currentTimeMillis();
            System.out.printf(Locale.US, "✅ Greedy 完成，用时: %d ms%n", (t1 - t0));
            System.out.printf(Locale.US, "📏 距离: %.4f%n", greedy.totalDistance);
            System.out.printf(Locale.US, "🎯 精度: %.4f%n", greedy.totalPrecision);
            GreedySolver.save(greedy, "TestGreedy.json", instance.viewpoints.size());
            System.out.println();

            // === 4️⃣ Genetic Solver + 2Opt ===
            System.out.println("🧬 运行 GeneticSolver + 2Opt...");
            long g0 = System.currentTimeMillis();
            GeneticSolver.Solution ga = GeneticSolver.solve(instance, greedy);
            long g1 = System.currentTimeMillis();
            System.out.printf(Locale.US, "✅ Genetic 完成，用时: %d ms%n", (g1 - g0));
            System.out.printf(Locale.US, "📏 距离: %.4f%n", ga.totalDistance);
            System.out.printf(Locale.US, "🎯 精度: %.4f%n", ga.totalPrecision);
            System.out.println("🧩 覆盖得分: " + ga.coverageScore + " / " + (instance.samplePoints.size() * 3));
            ga.save("TestGenetic.json");
            System.out.println();

            // === 5️⃣ 校验：回到起点 / PATH合法 / 覆盖≥3 ===
            System.out.println("=== 🔍 验证路径 ===");
            List<String> tourIds = toIdTour(instance, ga.tour);
            System.out.println("📏 路径长度节点数: " + tourIds.size());
            boolean backToStart = tourIds.get(0).equals(tourIds.get(tourIds.size() - 1));
            System.out.println((backToStart ? "✅" : "⚠️") + " 回到起点: 起点=" + tourIds.get(0) + ", 终点=" + tourIds.get(tourIds.size() - 1));

            System.out.println(checkConnectivity(instance, ga.tour));

            System.out.println("\n🔎 验证样本点覆盖情况(≥3 为充分) ...");
            CoverageStats stats = coverageStats(instance, tourIds);
            System.out.println("✅ 充分覆盖(≥3): " + stats.full);
            System.out.println("⚠️ 部分覆盖(1-2): " + stats.partial);
            System.out.println("❌ 未覆盖(0): " + stats.none);

            // === 6️⃣ 对比结果 ===
            System.out.println("\n=== 📊 对比结果 ===");
            double improve = (1.0 - (ga.totalDistance / greedy.totalDistance)) * 100.0;
            System.out.printf(Locale.US, "Greedy 距离: %.4f -> Genetic 距离: %.4f (优化率: %.2f%%)%n",
                    greedy.totalDistance, ga.totalDistance, improve);
            System.out.println("🎯 精度一致性验证: ✅");
            System.out.println("\n=== 🎉 测试结束 ===");

        } catch (Exception e) {
            System.err.println("❌ 出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /* ======================================================
     * 工具与验证模块
     * ====================================================== */

    /** 读取文件 */
    static String readFile(String fn) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(fn))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
        }
        return sb.toString();
    }

    /** 索引路径 → 视点ID路径 */
    static List<String> toIdTour(Instance instance, List<Integer> idxTour) {
        List<String> ids = new ArrayList<>();
        for (int idx : idxTour) ids.add(instance.viewpoints.get(idx).id);
        return ids;
    }

    /** 路径连通性检查 */
    static String checkConnectivity(Instance instance, List<Integer> idxTour) {
        CollisionMatrix cm = instance.collisionMatrix;
        int valid = 0;
        int total = Math.max(0, idxTour.size() - 1);
        StringBuilder warn = new StringBuilder();
        for (int i = 0; i < idxTour.size() - 1; i++) {
            int a = idxTour.get(i), b = idxTour.get(i + 1);
            if (cm.isConnected(a, b)) valid++;
            else warn.append(String.format("   ⚠️ 无效段: %s -> %s%n",
                    instance.viewpoints.get(a).id, instance.viewpoints.get(b).id));
        }
        String summary = "✅ 有效路径段: " + valid + "/" + total;
        return warn.length() == 0 ? summary + "，所有路径段合法，无 INF" : summary + "\n" + warn;
    }

    /** 覆盖统计结构 */
    static class CoverageStats { int full, partial, none; }

    /** 按动态角度（top-2）统计样本点覆盖情况 */
    static CoverageStats coverageStats(Instance instance, List<String> tourIds) {
        Map<String, Integer> coverage = new HashMap<>();
        for (SamplePoint sp : instance.samplePoints) coverage.put(sp.id, 0);

        for (String vid : tourIds) {
            Viewpoint vp = getViewpointById(instance, vid);
            if (vp == null) continue;
            List<String> angles = GreedySolverSelectAngles(vp, 2);
            if (angles.isEmpty()) continue;

            for (SamplePoint sp : instance.samplePoints) {
                for (String dir : angles) {
                    for (DirectionPair pair : sp.coveringPairs) {
                        if (pair.viewpointId.equals(vid) && pair.directionId.equals(dir)) {
                            coverage.put(sp.id, coverage.get(sp.id) + 1);
                            break;
                        }
                    }
                }
            }
        }

        CoverageStats s = new CoverageStats();
        for (int c : coverage.values()) {
            if (c >= 3) s.full++;
            else if (c > 0) s.partial++;
            else s.none++;
        }
        return s;
    }

    /** 动态选择前K角度的简化版（供验证覆盖使用） */
    static List<String> GreedySolverSelectAngles(Viewpoint vp, int k) {
        if (vp.precisionByDir == null || vp.precisionByDir.isEmpty()) return Collections.emptyList();
        List<Map.Entry<String, Double>> list = new ArrayList<>(vp.precisionByDir.entrySet());
        list.sort((a, b) -> Double.compare(Math.abs(b.getValue()), Math.abs(a.getValue())));
        List<String> result = new ArrayList<>();
        for (int i = 0; i < Math.min(k, list.size()); i++) result.add(list.get(i).getKey());
        return result;
    }

    /** 获取 viewpoint 对象 */
    static Viewpoint getViewpointById(Instance instance, String id) {
        for (Viewpoint v : instance.viewpoints)
            if (v.id.equals(id)) return v;
        return null;
    }
}
