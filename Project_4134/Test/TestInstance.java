import java.io.*;
import java.util.*;

/**
 * 测试类：读取 JSON 文件 → 解析 → 构建数据实例 → 打印结构化信息
 */
public class TestInstance {
    public static void main(String[] args) {
        try {
            // ========== Step 1: 从文件读取 JSON ==========
            String inputFile = "input.json"; // 你可以修改为实际路径
            System.out.println("[INFO] Reading file: " + inputFile);

            BufferedReader br = new BufferedReader(new FileReader(inputFile));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
            br.close();

            String jsonString = sb.toString();
            System.out.println("[INFO] JSON length: " + jsonString.length() + " chars");

            // ========== Step 2: 调用自定义 JsonParser ==========
            long t1 = System.currentTimeMillis();
            JsonParser parser = new JsonParser(jsonString);
            Map<String, Object> rawData = (Map<String, Object>) parser.parse();
            long t2 = System.currentTimeMillis();
            System.out.println("[INFO] ✅ JSON parsed successfully (" + (t2 - t1) + " ms)");

            // ========== Step 3: 构建 Instance ==========
            InstanceBuilder builder = new InstanceBuilder(rawData);
            Instance instance = builder.build();
            long t3 = System.currentTimeMillis();
            System.out.println("[INFO] ✅ Instance built (" + (t3 - t2) + " ms)");

            // ========== Step 4: 打印结构化信息 ==========
            printSummary(instance);

        } catch (Exception e) {
            System.err.println("[ERROR] Exception occurred during parsing/building:");
            e.printStackTrace();
        }
    }

    /**
     * 打印 Instance 的基本结构信息
     */
    private static void printSummary(Instance ins) {
        System.out.println("\n========== 数据解析结果 ==========");
        System.out.println("📍 视点数量 (viewpoints): " + ins.viewpoints.size());
        System.out.println("🎯 采样点数量 (sample_points): " + ins.samplePoints.size());
        System.out.println("🔢 碰撞矩阵维度: " + ins.collisionMatrix.size() + "x" + ins.collisionMatrix.size());
        System.out.println("🚀 起点视点 ID: " + ins.startViewpointId);
        System.out.println("=================================\n");

        // 打印前几个视点
        System.out.println("== 前 3 个视点示例 ==");
        for (int i = 0; i < Math.min(10, ins.viewpoints.size()); i++) {
            Viewpoint v = ins.viewpoints.get(i);
            System.out.printf("  [%d] %s 位置=(%.2f, %.2f, %.2f) 方向=%d\n",
                    i, v.id, v.x, v.y, v.z, v.precisionByDir.size());
        }

        // 打印前几个采样点
        System.out.println("\n== 前 3 个采样点示例 ==");
        for (int i = 0; i < Math.min(10, ins.samplePoints.size()); i++) {
            SamplePoint s = ins.samplePoints.get(i);
            System.out.printf("  [%d] %s 位置=(%.2f, %.2f, %.2f) 覆盖对=%d\n",
                    i, s.id, s.x, s.y, s.z, s.coveringPairs.size());
        }

        System.out.println("\n✅ 数据加载与结构验证完成。\n");
    }
}
