import java.util.List;
import java.util.*;
import java.io.*;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * ✅ 贪心路径规划算法
 * 从起点出发，每次选择最近的未访问 viewpoint，直到所有点都访问完。
 * 可在后续扩展为 2-opt 或遗传算法版本。
 */
class GreedySolver {

    /**
     * 主求解函数
     * @param ins 输入实例（包含 viewpoints / collisionMatrix / 起点）
     * @return Solution 对象，包含路径和总距离
     */
    public static Solution solve(Instance ins) {
        List<Viewpoint> vps = ins.viewpoints;
        int n = vps.size();
        if (n == 0) {
            System.err.println("[ERROR] No viewpoints to solve.");
            return new Solution(new ArrayList<>(), 0);
        }

        // === Step 1. 生成距离矩阵 ===
        double[][] dist = ins.computeDistanceMatrix();

        // === Step 2. 起点信息 ===
        String startId = ins.startViewpointId;
        int startIdx = findIndexById(vps, startId);
        if (startIdx == -1) startIdx = 0; // fallback

        boolean[] visited = new boolean[n];
        List<String> path = new ArrayList<>();
        int curr = startIdx;
        path.add(vps.get(curr).id);
        visited[curr] = true;

        // === Step 3. 贪心选最近的未访问点 ===
        for (int step = 1; step < n; step++) {
            int next = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                if (!visited[j] && dist[curr][j] < best) {
                    best = dist[curr][j];
                    next = j;
                }
            }
            if (next == -1 || best == Double.POSITIVE_INFINITY) break; // 无法到达剩余点
            visited[next] = true;
            path.add(vps.get(next).id);
            curr = next;
        }

        // === Step 4. 计算总路径长度 ===
        double totalDist = 0;
        for (int i = 0; i < path.size() - 1; i++) {
            int a = findIndexById(vps, path.get(i));
            int b = findIndexById(vps, path.get(i + 1));
            totalDist += dist[a][b];
        }

        System.out.println("[INFO] Greedy path length = " + totalDist);
        System.out.println("[INFO] Path visited = " + path.size() + " viewpoints");

        return new Solution(path, totalDist);
    }

    // 工具函数：根据 id 找索引
    private static int findIndexById(List<Viewpoint> vps, String id) {
        for (int i = 0; i < vps.size(); i++) {
            if (vps.get(i).id.equals(id)) return i;
        }
        return -1;
    }
}




/**
 * 表示路径解（Solution）
 * 输出格式兼容老师提供的 solution_sample.json
 */
class Solution {
    List<String> path;       // viewpoint ID 顺序
    double totalDistance;    // 总距离（目标函数值）

    public Solution(List<String> path, double totalDistance) {
        this.path = path;
        this.totalDistance = totalDistance;
    }

    /**
     * 转换为老师要求的 JSON 格式
     */
    public String toJsonFormatted() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"metadata\": {\n");
        sb.append("    \"num_viewpoints\": ").append(path.size()).append(",\n");
        sb.append("    \"objective\": {\n");
        sb.append("      \"distance\": ").append(String.format(Locale.US, "%.6f", totalDistance)).append(",\n");
        sb.append("      \"precision\": 0.0\n");
        sb.append("    }\n");
        sb.append("  },\n");
        sb.append("  \"sequence\": [\n");

        for (int i = 0; i < path.size(); i++) {
            sb.append("    { \"id\": \"").append(path.get(i))
                    .append("\", \"angles\": [\"a1\", \"a3\"] }");
            if (i != path.size() - 1) sb.append(",");
            sb.append("\n");
        }

        sb.append("  ]\n");
        sb.append("}");
        return sb.toString();
    }

    /**
     * 保存到文件（UTF-8）
     */
    public void saveToFile(String filename) {
        try (PrintWriter out = new PrintWriter(new FileWriter(filename, false))) {
            out.println(toJsonFormatted());
            System.out.println("[INFO] Solution saved to " + filename);
        } catch (IOException e) {
            System.err.println("[ERROR] Failed to save solution: " + e.getMessage());
        }
    }

    @Override
    public String toString() {
        return "Solution{path=" + path.size() + " points, totalDistance=" + totalDistance + "}";
    }
}


