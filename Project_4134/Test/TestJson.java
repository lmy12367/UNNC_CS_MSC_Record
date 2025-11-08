import java.io.*;
import java.util.*;

/**
 * JSON解析器测试类
 * 功能：读取JSON文件并打印解析结果
 */
public class TestJson {
    public static void main(String[] args) {
        try {
            System.out.println("=== JSON解析器测试 ===\n");

            // 1. 读取JSON文件
            System.out.println("1. 读取input.json文件...");
            String jsonContent = readFile("input.json");
            System.out.println("✅ 文件读取成功，长度: " + jsonContent.length() + " 字符\n");

            // 2. 解析JSON
            System.out.println("2. 开始解析JSON...");
            long startTime = System.currentTimeMillis();
            JsonParser parser = new JsonParser(jsonContent);
            Object result = parser.parse();
            long endTime = System.currentTimeMillis();
            System.out.println("✅ 解析完成，耗时: " + (endTime - startTime) + " 毫秒\n");

            // 3. 打印解析结果
            System.out.println("3. 解析结果结构:");
            System.out.println("========================");
            printResult(result, 0);
            System.out.println("========================\n");

            // 4. 显示统计信息
            showStatistics(result);

        } catch (Exception e) {
            System.err.println("❌ 测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 读取文件内容
     */
    private static String readFile(String filename) throws IOException {
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line);
            }
        }
        return content.toString();
    }

    /**
     * 递归打印解析结果
     */
    private static void printResult(Object obj, int indent) {
        String prefix = "  ".repeat(indent);  // 缩进字符串

        if (obj instanceof Map) {
            // 打印对象
            System.out.println(prefix + "{");
            Map<?, ?> map = (Map<?, ?>) obj;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                System.out.print(prefix + "  \"" + entry.getKey() + "\": ");
                printResult(entry.getValue(), indent + 1);
                System.out.println();
            }
            System.out.print(prefix + "}");

        } else if (obj instanceof List) {
            // 打印数组
            System.out.println(prefix + "[");
            List<?> list = (List<?>) obj;
            for (int i = 0; i < list.size(); i++) {
                System.out.print(prefix + "  [" + i + "]: ");
                printResult(list.get(i), indent + 1);
                if (i < list.size() - 1) System.out.println();
            }
            System.out.print(prefix + "]");

        } else if (obj instanceof String) {
            // 打印字符串（带引号）
            System.out.print("\"" + obj + "\"");

        } else {
            // 打印数字、布尔值、null
            System.out.print(obj + " (" + obj.getClass().getSimpleName() + ")");
        }
    }

    /**
     * 显示解析统计信息
     */
    private static void showStatistics(Object obj) {
        System.out.println("4. 统计信息:");
        System.out.println("   - 根类型: " + obj.getClass().getSimpleName());

        if (obj instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) obj;
            System.out.println("   - 对象字段数: " + map.size());
            System.out.println("   - 主要字段:");
            for (Object key : map.keySet()) {
                System.out.println("     * " + key + " (" + map.get(key).getClass().getSimpleName() + ")");
            }
        }

        System.out.println("\n=== 测试完成 ===");
    }
}
