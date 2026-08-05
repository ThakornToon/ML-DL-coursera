package ML;

import ML.tools.Dense;
import ML.tools.Sequential;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Implementation of the Deep Q-Learning - Lunar Lander Assignment (C3W3A1).
 *
 * <p>This assignment focuses on Reinforcement Learning (RL) concepts, primarily: 1. Building a
 * Q-Network and Target Q-Network. 2. Calculating the loss using the Bellman Equation
 * (compute_loss). 3. Soft updating the Target Q-Network weights. 4. Experience Replay and
 * Epsilon-Greedy training loops.
 */
/**
 * Implementation of the C3W3 Assignment 1: Deep Q-Learning.
 * Demonstrates Reinforcement Learning principles with Q-networks and experience replay.
 */
public final class C3W3A1Assignment {

    private C3W3A1Assignment() {}

    /**
     * Executes the Deep Q-Learning assignment demonstration.
     */
    public static void start() {
        System.out.println("Deep Q-Learning - Lunar Lander (C3W3A1) - Full Training Loop");

        int stateSize = 8;
        int numActions = 4;

        // Create the Q-Network and Target Q^-Network
        Sequential qNetwork = new Sequential();
        qNetwork.add(new Dense(64, "relu", "q_layer1"));
        qNetwork.add(new Dense(64, "relu", "q_layer2"));
        qNetwork.add(new Dense(numActions, "linear", "q_layer3"));

        Sequential targetQNetwork = new Sequential();
        targetQNetwork.add(new Dense(64, "relu", "target_layer1"));
        targetQNetwork.add(new Dense(64, "relu", "target_layer2"));
        targetQNetwork.add(new Dense(numActions, "linear", "target_layer3"));

        // Initialize weights by passing dummy data
        double[][] dummyData = new double[1][stateSize];
        qNetwork.predict(dummyData);
        targetQNetwork.predict(dummyData);

        boolean loadModel = true; // Change to true to load existing model weights

        if (loadModel) {
            System.out.println("Loading saved weights...");
            for (ML.tools.Dense layer : qNetwork.getLayers()) {
                String filename = "ml_c3w3a1_assign_" + layer.getName() + "_weights.txt";
                Object[] loaded = ML.tools.ModelWeightsIO.loadDenseWeights(filename);
                if (loaded != null) layer.setWeights((double[][]) loaded[0], (double[]) loaded[1]);
            }
            for (ML.tools.Dense layer : targetQNetwork.getLayers()) {
                String filename = "ml_c3w3a1_assign_" + layer.getName() + "_weights.txt";
                Object[] loaded = ML.tools.ModelWeightsIO.loadDenseWeights(filename);
                if (loaded != null) layer.setWeights((double[][]) loaded[0], (double[]) loaded[1]);
            }
        }

        // Initialize target network with exact weights of Q-Network (tau = 1.0)
        updateTargetNetwork(qNetwork, targetQNetwork, 1.0);

        LunarLanderEnv env = new LunarLanderEnv();
        ReplayBuffer buffer = new ReplayBuffer(2000); // Small buffer for memory efficiency in Java

        // Hyperparameters
        int numEpisodes = 500;
        int maxStepsPerEpisode = 300;
        int batchSize = 64;
        double gamma = 0.995;
        double alpha = 1e-3; // learning rate
        double tau = 1e-3;
        int updateEvery = 4;

        double epsilon = 1.0;
        double epsilonMin = 0.01;
        double epsilonDecay = 0.995;

        int t = 1; // Optimizer step counter

        java.util.LinkedList<Double> rewardHistory = new java.util.LinkedList<>();

        System.out.println("Starting training for " + numEpisodes + " episodes...");

        for (int episode = 1; episode <= numEpisodes; episode++) {
            // Start random initial state
            double[] state = env.reset();
            double totalReward = 0;

            for (int step = 0; step < maxStepsPerEpisode; step++) {
                // Epsilon-Greedy action selection
                int action = epsilonGreedy(state, epsilon, numActions, qNetwork);

                // Step environment
                StepResult result = env.step(action);

                // Store experience
                buffer.add(state, action, result);

                totalReward += result.reward;
                state = result.nextState.clone();

                // Learn every C steps
                if ((step + 1) % updateEvery == 0 && buffer.size() >= batchSize) {
                    agentLearn(buffer, batchSize, gamma, qNetwork, targetQNetwork, alpha, t);
                    t++;

                    // Soft update target network
                    updateTargetNetwork(qNetwork, targetQNetwork, tau);
                }

                if (result.done) {
                    break;
                }
            }

            // Epsilon decay
            epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);

            // Track moving average
            rewardHistory.add(totalReward);
            if (rewardHistory.size() > 100) {
                rewardHistory.removeFirst();
            }

            // Average reward from recent
            double movingAvg = 0;
            for (Double r : rewardHistory) movingAvg += r;
            movingAvg /= rewardHistory.size();

            if (episode % 50 == 0 || episode == 1) {
                System.out.printf("Episode %d \t Total Reward: %8.2f \t Avg Reward (last 100): %8.2f \t Epsilon: %.3f%n", episode, totalReward, movingAvg, epsilon);
            }

            // Stop early if solved
            if (movingAvg >= 100.0 && rewardHistory.size() >= 100) {
                System.out.println("\nEnvironment solved in " + episode + " episodes!");
                break;
            }
        }

        System.out.println("Saving model weights...");
        for (ML.tools.Dense layer : qNetwork.getLayers()) {
            if (layer.isInitialized()) {
                String filename = "ml_c3w3a1_assign_" + layer.getName() + "_weights.txt";
                ML.tools.ModelWeightsIO.saveWeights(filename, layer.getWeightsW(), layer.getWeightsB());
            }
        }
        for (ML.tools.Dense layer : targetQNetwork.getLayers()) {
            if (layer.isInitialized()) {
                String filename = "ml_c3w3a1_assign_" + layer.getName() + "_weights.txt";
                ML.tools.ModelWeightsIO.saveWeights(filename, layer.getWeightsW(), layer.getWeightsB());
            }
        }

        System.out.println("Training complete.");
        System.out.println("=================================================================\n");
    }

    /**
     * agentLearn executes the forward pass, custom backpropagation for multi-output MSE, and updates
     * the weights using the Adam optimizer.
     */
    public static double agentLearn(
            ReplayBuffer buffer,
            int batchSize,
            double gamma,
            Sequential qNetwork,
            Sequential targetQNetwork,
            double learningRate,
            int t) {

        Object[] batch = buffer.sample(batchSize);  // Get some batch from buffer

        // Data all batch
        double[][] states = (double[][]) batch[0];
        int[] actions = (int[]) batch[1];
        double[] rewards = (double[]) batch[2];
        double[][] nextStates = (double[][]) batch[3];
        double[] doneVals = (double[]) batch[4];

        int m = batchSize;
        int numActions = 4;

        // 1. Calculate target y values (Bellman) predict from targetQNetwork of all batch nextStates
        double[][] targetQValuesAll = targetQNetwork.predict(nextStates);
        double[] yTargets = new double[m];
        for (int i = 0; i < m; i++) {
            double maxVal = -Double.MAX_VALUE;
            for (int j = 0; j < numActions; j++) {
                if (targetQValuesAll[i][j] > maxVal) {
                    maxVal = targetQValuesAll[i][j];
                }
            }
            yTargets[i] = rewards[i] + (gamma * maxVal * (1.0 - doneVals[i]));
        }

        // 2. Forward pass qNetwork saving internal caches
        double[][] A = states;
        for (Dense layer : qNetwork.getLayers()) {
            A = layer.forwardBatch(A);
        }
        double[][] qValuesAll = A;

        // 3. Compute loss and dA (derivative of Loss with respect to Output)
        double[][] dA = new double[m][numActions];
        double loss = 0;

        for (int i = 0; i < m; i++) {
            int a = actions[i];
            double diff = qValuesAll[i][a] - yTargets[i]; // Derivative of 0.5*(Q - Y)^2
            loss += diff * diff;

            // Set derivative only for the action taken, gradients for other actions remain 0
            dA[i][a] = diff;
        }
        loss /= m;

        // 4. Backward pass
        for (int l = qNetwork.getLayers().size() - 1; l >= 0; l--) {
            dA = qNetwork.getLayers().get(l).backward(dA, learningRate, "adam", t);
        }

        return loss;
    }

    /** Selects an action using Epsilon-Greedy policy. */
    public static int epsilonGreedy(double[] state, double epsilon, int numActions, Sequential qNetwork) {
        if (Math.random() < epsilon) { // Random action
            return (int) (Math.random() * numActions);
        } else { // Use qNetwork to predict and select action
            double[][] qValues = qNetwork.predict(new double[][] {state});
            double maxQ = -Double.MAX_VALUE;
            int bestAction = 0;
            // For all action
            for (int i = 0; i < numActions; i++) {
                if (qValues[0][i] > maxQ) {
                    maxQ = qValues[0][i];
                    bestAction = i;
                }
            }
            return bestAction;
        }
    }

    /** Soft updates the weights of the target network. w_target = tau * w_q + (1 - tau) * w_target */
    public static void updateTargetNetwork(Sequential qNetwork, Sequential targetQNetwork, double tau) {
        for (int i = 0; i < qNetwork.getLayers().size(); i++) {
            Dense qLayer = qNetwork.getLayers().get(i);
            Dense targetLayer = targetQNetwork.getLayers().get(i);

            double[][] qW = qLayer.getWeightsW();
            double[] qB = qLayer.getWeightsB();

            double[][] tW = targetLayer.getWeightsW();
            double[] tB = targetLayer.getWeightsB();

            if (qW == null) continue;

            double[][] newTW = new double[qW.length][qW[0].length];
            double[] newTB = new double[qB.length];

            if (tW == null || !targetLayer.isInitialized()) {
                // Initial copy
                for (int k = 0; k < qW.length; k++) {
                    System.arraycopy(qW[k], 0, newTW[k], 0, qW[k].length);
                }
                System.arraycopy(qB, 0, newTB, 0, qB.length);
            } else {
                // Soft update
                for (int k = 0; k < qW.length; k++) {
                    for (int j = 0; j < qW[k].length; j++) {
                        newTW[k][j] = tau * qW[k][j] + (1.0 - tau) * tW[k][j];
                    }
                }
                for (int j = 0; j < qB.length; j++) {
                    newTB[j] = tau * qB[j] + (1.0 - tau) * tB[j];
                }
            }
            targetLayer.setWeights(newTW, newTB);
        }
    }

    /** Replay buffer to store and sample experiences. */
    public static class ReplayBuffer {
        private int capacity;
        private List<StepResult> buffer;
        private List<double[]> states;
        private List<Integer> actions;
        private Random random = new Random();

        public ReplayBuffer(int capacity) {
            this.capacity = capacity;
            this.buffer = new ArrayList<>();
            this.states = new ArrayList<>();
            this.actions = new ArrayList<>();
        }

        public void add(double[] state, int action, StepResult result) {
            if (buffer.size() >= capacity) {
                buffer.remove(0);
                states.remove(0);
                actions.remove(0);
            }
            buffer.add(result);
            states.add(state.clone());
            actions.add(action);
        }

        public int size() {
            return buffer.size();
        }

        public Object[] sample(int batchSize) {
            int stateSize = states.get(0).length;
            double[][] s = new double[batchSize][stateSize]; // state
            int[] a = new int[batchSize]; // action
            double[] r = new double[batchSize]; // reward
            double[][] ns = new double[batchSize][stateSize]; // next state
            double[] d = new double[batchSize]; // done value

            // Random each batch from buffer
            for (int i = 0; i < batchSize; i++) {
                int idx = random.nextInt(buffer.size()); // random index
                s[i] = states.get(idx);
                a[i] = actions.get(idx);
                StepResult sr = buffer.get(idx);
                r[i] = sr.reward;
                ns[i] = sr.nextState;
                d[i] = sr.done ? 1.0 : 0.0;
            }
            return new Object[] {s, a, r, ns, d};
        }
    }

    /** Data structure to hold the result of an environment step. */
    public static class StepResult {
        public double[] nextState;
        public double reward;
        public boolean done;

        public StepResult(double[] nextState, double reward, boolean done) {
            this.nextState = nextState;
            this.reward = reward;
            this.done = done;
        }
    }

    /**
     * A simplified Java simulation of the Gym LunarLander-v2 environment. Generates physically
     * plausible (though much simpler) state transitions and rewards.
     */
    public static class LunarLanderEnv {
        private double[] state;

        public double[] reset() {
            state = new double[8];
            state[0] = (Math.random() - 0.5) * 0.2; // x
            state[1] = 1.4 + Math.random() * 0.2; // y
            state[2] = (Math.random() - 0.5) * 0.1; // vx
            state[3] = (Math.random() - 0.5) * 0.1; // vy
            state[4] = (Math.random() - 0.5) * 0.1; // angle
            state[5] = (Math.random() - 0.5) * 0.1; // angular velocity
            state[6] = 0.0; // left leg
            state[7] = 0.0; // right leg
            return state.clone();
        }

        public StepResult step(int action) {
            double x = state[0];
            double y = state[1];
            double vx = state[2];
            double vy = state[3];
            double angle = state[4];
            double vAngle = state[5];

            double reward = 0.0;

            // 1. Apply action forces
            vy -= 0.005; // gravity pulling down
            if (action == 1) { // right engine fires (pushes lander left)
                vx -= 0.002;
                vAngle -= 0.002;
                reward -= 0.03;
            } else if (action == 2) { // main engine fires (pushes lander up)
                vy += 0.008;
                reward -= 0.3;
            } else if (action == 3) { // left engine fires (pushes lander right)
                vx += 0.002;
                vAngle += 0.002;
                reward -= 0.03;
            }

            // 2. Update kinematics
            x += vx;
            y += vy;
            angle += vAngle;

            // 3. Compute base rewards (penalty for being far from center pad (0,0))
            double dist = Math.sqrt(x * x + y * y);
            reward -= 0.1 * dist;

            // 4. Check for episode termination (crash or land)
            boolean done = false;
            if (y <= 0.0) {
                done = true;
                // Simple landing condition: slow speed, upright, near center
                if (Math.abs(angle) < 0.2 && Math.abs(vy) < 0.05 && Math.abs(x) < 0.2) {
                    reward += 100.0; // Safe landing
                    state[6] = 1.0; // Legs contact
                    state[7] = 1.0;
                } else {
                    reward -= 100.0; // Crash
                }
                y = 0.0;
            } else if (x < -1.0 || x > 1.0 || y > 2.0) {
                done = true;
                reward -= 100.0; // Out of bounds
            }

            // Update state
            state[0] = x;
            state[1] = y;
            state[2] = vx;
            state[3] = vy;
            state[4] = angle;
            state[5] = vAngle;

            return new StepResult(state.clone(), reward, done);
        }
    }
}
