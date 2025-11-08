import java.util.Scanner;
/*
 * @author Mingyuan LIU
 * @version 3.0
 * @since 2025-10-24
 * @StudentID:20808220
 * @Email:scxml3@nottingham.edu.cn
 */

/**
 * Node class for the AVL tree.
 * <p>
 * Each node stores a value, references to its left and right children, and the height of the node.
 * </p>
 */
class TreeNode {
    public int val;        // The value stored in the node
    public TreeNode left;  // Left child node
    public TreeNode right; // Right child node
    public int height;     // Height of the subtree rooted at this node

    /**
     * Constructs a new tree node.
     *
     * @param val The integer value to be stored in the node.
     */
    public TreeNode(int val) {
        this.val = val;
        // New node's initial height is 1 (leaf node)
        this.height = 1;
    }
}
/**
 * Represents a self-balancing binary tree with custom insertion rules.
 * <p>
 * This tree adds an interesting rule based on the AVL tree:
 * <ul>
 *   <li>When inserting an <strong>even</strong> number, it always tries to insert it into the <strong>right subtree</strong>
 *       of the current node. If the right subtree is empty, it is inserted directly; if not, the process continues recursively into the right subtree.</li>
 *   <li>When inserting an <strong>odd</strong> number, it follows the standard Binary Search Tree (BST) insertion rules.</li>
 * </ul>
 * Although the insertion rule might violate the properties of a Binary Search Tree, after each insertion, the tree is rebalanced
 * using standard AVL rotation operations (left rotation, right rotation, and their combinations) to restore height balance.
 * Therefore, it is a "non-BST AVL tree".
 * </p>
 */
public class AVL {
    // The root node of the AVL tree
    private TreeNode root;

    /**
     * Gets the root node of the current AVL tree.
     *
     * @return The root of the tree, or null if the tree is empty.
     */
    private TreeNode getRoot() {
        return this.root;
    }

    /**
     * Gets the height of a given node.If the node is null, its height is 0.
     *
     * @param node The node whose height is to be calculated.
     * @return The height of the node.
     */
    private int getHeight(TreeNode node) {
        if (node == null) {
            return 0;
        }
        return node.height;
    }

    /**
     * Calculates the balance factor of a given node.
     * <p>
     * Balance Factor = height of left subtree - height of right subtree.
     * For an AVL tree, the balance factor of any node must be -1, 0, or 1.
     * </p>
     *
     * @param node The node for which to calculate the balance factor.
     * @return The balance factor.
     */
    private int getBalance(TreeNode node) {
        if (node == null) {
            return 0;
        }
        return getHeight(node.left) - getHeight(node.right);
    }

    /**
     * Performs a right rotation on the subtree rooted at pivot.
     * <p>
     * Right rotation is used to handle the "Left-Left" (LL) imbalance case.
     * Before rotation:          After rotation:
     *       pivot            newRoot
     *      /     \          /       \
     *  newRoot  T3   -->   T1     pivot
     *    /     \                  /     \
     *   T1   middleSubtree    middleSubtree  T3
     * </p>
     *
     * @param pivot The root node of the unbalanced subtree.
     * @return The new root of the subtree after rotation.
     */
    private TreeNode rightRotate(TreeNode pivot) {
        TreeNode newRoot = pivot.left;
        TreeNode middleSubtree = newRoot.right;

        // Perform rotation
        newRoot.right = pivot;
        pivot.left = middleSubtree;

        // Update heights
        pivot.height = Math.max(getHeight(pivot.left), getHeight(pivot.right)) + 1;
        newRoot.height = Math.max(getHeight(newRoot.left), getHeight(newRoot.right)) + 1;

        // Return the new root
        return newRoot;
    }

    /**
     * Performs a left rotation on the subtree rooted at pivot.
     * <p>
     * Left rotation is used to handle the "Right-Right" (RR) imbalance case.
     * Before rotation:          After rotation:
     *       pivot            newRoot
     *      /     \          /       \
     *     T1  newRoot   --> pivot     T3
     *          /     \      /     \
     * middleSubtree  T3   T1  middleSubtree
     * </p>
     *
     * @param pivot The root node of the unbalanced subtree.
     * @return The new root of the subtree after rotation.
     */
    private TreeNode leftRotate(TreeNode pivot) {
        TreeNode newRoot = pivot.right;
        TreeNode middleSubtree = newRoot.left;

        // Perform rotation
        newRoot.left = pivot;
        pivot.right = middleSubtree;

        // Update heights
        pivot.height = Math.max(getHeight(pivot.left), getHeight(pivot.right)) + 1;
        newRoot.height = Math.max(getHeight(newRoot.left), getHeight(newRoot.right)) + 1;

        // Return the new root
        return newRoot;
    }

    /**
     * Inserts a new value into the AVL tree.
     * <p>
     * This is the public interface, which calls the private recursive insertion method.
     * </p>
     *
     * @param val The integer value to insert.
     */
    public void insert(int val) {
        this.root = insert(this.root, val);
    }

    /**
     * Recursively inserts a value into a subtree of the AVL tree and ensures the tree remains balanced.
     * <p>
     * This method implements the custom insertion rules:
     * <ul>
     *   <li>Even numbers: force insertion into the right subtree.</li>
     *   <li>Odd numbers: insert according to standard BST rules.</li>
     * </ul>
     * After insertion, the rotation type is determined by checking the balance factors of the subtrees,
     * which is more robust than comparing the inserted value.
     * </p>
     *
     * @param node The root of the current subtree.
     * @param val  The value to insert.
     * @return The root of the updated and balanced subtree after inserting the new value.
     */
    private TreeNode insert(TreeNode node, int val) {
        // 1. Perform custom insertion logic
        if (node == null) {
            // Base case for recursion: create a new node.
            return new TreeNode(val);
        }

        if (val % 2 == 0) {
            // Rule: even numbers always try to be inserted in the right subtree
            node.right = insert(node.right, val);
        }
        else {
            // Rule: odd numbers follow standard BST insertion rules
            if (val < node.val) {
                node.left = insert(node.left, val);
            }
            else if (val > node.val) {
                node.right = insert(node.right, val);
            }
            else {
                // Duplicate values are not allowed, return the current node directly
                return node;
            }
        }

        // 2. Update the height of the current node
        node.height = 1 + Math.max(getHeight(node.left), getHeight(node.right));

        // 3. Get the balance factor of the current node to check if it's balanced
        int balance = getBalance(node);

        // 4. If unbalanced, determine the rotation type based on the balance factors of the subtrees
        // This approach does not rely on comparing the inserted value, making it more robust for custom insertion rules

        // Case 1: Node is left-heavy (balance > 1)
        if (balance > 1) {
            // Subcase LL: Left subtree is also left-heavy or balanced
            if (getBalance(node.left) >= 0) {
                return rightRotate(node);
            }
            // Subcase LR: Left subtree is right-heavy
            else { // getBalance(node.left) < 0
                node.left = leftRotate(node.left);
                return rightRotate(node);
            }
        }

        // Case 2: Node is right-heavy (balance < -1)
        if (balance < -1) {
            // Subcase RR: Right subtree is also right-heavy or balanced
            if (getBalance(node.right) <= 0) {
                return leftRotate(node);
            }
            // Subcase RL: Right subtree is left-heavy
            else { // getBalance(node.right) > 0
                node.right = rightRotate(node.right);
                return leftRotate(node);
            }
        }

        // If the node is balanced, return it directly
        return node;
    }

    /**
     * Performs a post-order traversal (Left-Right-Root) of the tree.
     * <p>
     * This method recursively visits the left subtree of a node, then the right subtree, and finally the node itself.
     * The value of each node will be printed on a new line.
     * </p>
     *
     * @param node The starting node for the traversal. If null, no action is performed.
     */
    private static void postOrderTraversal(TreeNode node) {
        if (node != null) {
            postOrderTraversal(node.left);
            postOrderTraversal(node.right);
            System.out.println(node.val);
        }
    }

    /**
     * The main entry point for the program.
     * <p>
     * This method is responsible for reading a sequence of integers from standard input, building an AVL tree with custom rules,
     * and then calling the post-order traversal method to print the final result.
     * </p>
     */
    public static void main(String[] args) {
        // Create an AVL tree instance.
        AVL tree = new AVL();

        // Read input and build the tree using try-with-resources.
        try (Scanner scanner = new Scanner(System.in)) {
            // Set delimiter to parse comma/space-separated values.
            scanner.useDelimiter("[,\\s]+");

            while (scanner.hasNextInt()) {
                int value = scanner.nextInt();
                tree.insert(value);
            }
        } // Scanner is automatically closed here.

        // Print the tree using post-order traversal.
        postOrderTraversal(tree.getRoot());
    }
}

