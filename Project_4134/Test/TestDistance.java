import java.io.*;
import java.util.*;

public class TestDistance {
    public static void main(String[] args) throws Exception {
        // === Step 1: 读取 JSON 文件 ===
        String filePath = "input.json"; // 请改为你的实际路径
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
        }

        // === Step 2: 解析 JSON ===
        JsonParser parser = new JsonParser(sb.toString());
        @SuppressWarnings("unchecked")
        Map<String, Object> raw = (Map<String, Object>) parser.parse();

        // === Step 3: 构建实例 ===
        InstanceBuilder builder = new InstanceBuilder(raw);
        Instance instance = builder.build();

        // === Step 4: 计算距离矩阵 ===
        double[][] dist = instance.computeDistanceMatrix();

        System.out.println("[INFO] Distance matrix size: " + dist.length + "x" + dist[0].length);

        // === Step 5: 打印前 3x3 ===
        int limit = Math.min(3, dist.length);
        for (int i = 0; i < limit; i++) {
            for (int j = 0; j < limit; j++) {
                if (Double.isInfinite(dist[i][j])) {
                    System.out.print(" INF ");
                } else {
                    System.out.printf("%8.2f ", dist[i][j]);
                }
            }
            System.out.println();
        }
    }
}
