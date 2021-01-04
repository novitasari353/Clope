package weka.clusterers;

import java.io.Serializable; // for occurrence

import weka.core.*;
import java.util.*;

public class Clope extends AbstractClusterer implements OptionHandler {
    public ArrayList<ClopeCluster> clusters = new ArrayList<ClopeCluster>();
    public ArrayList<Integer> clusterWrite = new ArrayList();
    public double repulsionDefault;
    public double repulsionSelected;
    public int numberOfClusters;
    public boolean numberOfClustersDet;
    public int temp_instance;
    public int numberOfInstances;

    public Clope(){
        repulsionDefault = 2.6;
        repulsionSelected = repulsionDefault;
        numberOfClusters = 0;
        numberOfClustersDet = false;
    }

    public String globalInfo() {
        return "CLOPE: A Fast and Effective Clustering Algorithm for Transactional Data";
    }

    /**
     * Returns default capabilities of the clusterer.
     *
     * @return the capabilities of this clusterers
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // class
        result.enable(Capabilities.Capability.NO_CLASS);

        // missing value
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // attributes (nominal)
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        return result;
    }

    /**
     * Parses a given list of options.
     * <p/>
     *
     <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -R
     *  Set repulsion, integer.
     *  (default 2.6)</pre>
     *
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String repulsion = Utils.getOption('R', options);
        if (repulsion.length() != 0) {
            setRepulsion(Double.parseDouble(repulsion));
        } else {
            setRepulsion(repulsionDefault);
        }
    }

    /**
     * Gets the current settings of Clope
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<String>();

        options.add("-R");
        options.add("" + getRepulsion());
        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[1]);
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>();
        newVector.add(new Option("\tThe repulsion (default = 2.6).", "R", 1, "-R <num>"));
        return newVector.elements();
    }

    /**
     * Generates a predictor.
     *
     * @param argv the options
     */
    public static void main(String[] argv) {
        runClusterer(new Clope(), argv);
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        numberOfInstances = data.numInstances();
        boolean moved;

        for (int i = 0; i < data.numInstances(); i++) {
            int clusterNum = PutInstanceMaxProfit(data.instance(i));
            clusterWrite.add(clusterNum);
        }

        do {
            moved = false;
            for (int i = 0; i < data.numInstances(); i++) {
                temp_instance = i;
                int clusterNum = MoveInstanceMaxProfit(data.instance(i));
                if (clusterNum != clusterWrite.get(i)) {
                    clusterWrite.set(i, clusterNum);
                    moved = true;
                }
            }
        }
        while (moved = false);
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        if (temp_instance >= numberOfInstances) {
            temp_instance = 0;
        }
        int i = clusterWrite.get(temp_instance);
        temp_instance++;
        return i;
    }

    private class ClopeCluster implements Serializable {
        public int S = 0; // size
        public int W = 0; // width
        public int N = 0; // transaction
        public int count; // ++

        public HashMap occ = new HashMap(); // <item, occurrence>

        public double DeltaAdd(Instance ins, double r) {
            int S_new = 0;
            int W_new = occ.size();
            double profit;
            double profit_new;
            double delta_profit;

            for (int i = 0; i < ins.numAttributes(); i++) {
                if (!ins.isMissing(i)) {
                    S_new++;
                    if ((Integer) occ.get(i + ins.toString(i)) == null) {
                        W_new++;
                    }
                }
            }
            S_new += S;
            profit = S * N / Math.pow(W, r);
            profit_new = S_new * (N + 1) / Math.pow(W_new, r);

            delta_profit = profit_new - profit;
            return delta_profit;
        }

        public void addInstance(Instance ins) {
            for (int i = 0; i < ins.numAttributes(); i++) {
                if (!ins.isMissing(i)) {
                    if (!occ.containsKey(i + ins.toString(i))) {
                        occ.put(i + ins.toString(i), 1);
                    }
                    else {
                        count = (Integer) occ.get(i + ins.toString(i));
                        count++;
                        occ.remove(i + ins.toString(i));
                        occ.put(i + ins.toString(i), count);
                    }
                    S++;
                }
            }
            W = occ.size();
            N++;
        }

        public void deleteInstance(Instance ins) {
            for (int i = 0; i <= ins.numAttributes() - 1; i++) {
                if (!ins.isMissing(i)) {
                    count = (Integer) occ.get(i + ins.toString(i));
                    if (count == 1) {
                        occ.remove(i + ins.toString(i));
                    }
                    else {
                        count--;
                        occ.remove(i + ins.toString(i));
                        occ.put(i + ins.toString(i), count);
                    }
                    S--;
                }
            }
            W = occ.size();
            N--;
        }
    }

    public int numberOfClusters() {
        numberOfClusters = clusters.size();
        numberOfClustersDet = true;
        return numberOfClusters;
    }

    /**
     * Put instance that maximize profit
     */
    public int PutInstanceMaxProfit(Instance ins) {
        int temp_S = 0;
        int temp_W = 0;
        double delta;
        double delta_max;
        int cluster_max = 0;
        if (clusters.size() > 0) {
            for (int i = 0; i < ins.numAttributes(); i++) {
                if (!ins.isMissing(i)) {
                    temp_S++;
                    temp_W++;
                }
            }

            delta_max = temp_S/Math.pow(temp_W, repulsionSelected);

            for (int i = 0; i < clusters.size(); i++) {
                ClopeCluster temp_cluster = clusters.get(i);
                delta = temp_cluster.DeltaAdd(ins, repulsionSelected);
                if (delta > delta_max) {
                    delta_max = delta;
                    cluster_max = i;
                }
            }
        }
        else {
            ClopeCluster newCluster = new ClopeCluster();
            clusters.add(newCluster);
            newCluster.addInstance(ins);
            return clusters.size() - 1;
        }

        if (cluster_max == 0) {
            ClopeCluster newCluster = new ClopeCluster();
            clusters.add(newCluster);
            newCluster.addInstance(ins);
            return clusters.size() - 1;
        }
        else {
            clusters.get(cluster_max).addInstance(ins);
        }
        return cluster_max;
    }

    /**
     * Move instance that maximize profit
     */
    public int MoveInstanceMaxProfit(Instance ins) {
        clusters.get(clusterWrite.get(temp_instance)).deleteInstance(ins);
        clusterWrite.set(temp_instance, -1);
        double delta;
        double delta_max;
        int cluster_max = 0;
        int temp_S = 0;
        int temp_W = 0;

        for (int i = 0; i < ins.numAttributes(); i++) {
            if (!ins.isMissing(i)) {
                temp_S++;
                temp_W++;
            }
        }

        delta_max = temp_S / Math.pow(temp_W, repulsionSelected);
        for (int i = 0; i < clusters.size(); i++) {
            ClopeCluster temp_cluster = clusters.get(i);
            delta = temp_cluster.DeltaAdd(ins, repulsionSelected);
            if (delta > delta_max) {
                delta_max = delta;
                cluster_max = i;
            }
        }

        if (cluster_max == 0) {
            ClopeCluster newCluster = new ClopeCluster();
            clusters.add(newCluster);
            newCluster.addInstance(ins);
            return clusters.size() - 1;
        }
        else {
            clusters.get(cluster_max).addInstance(ins);
        }
        return cluster_max;
    }

    public double getRepulsion() {
        return repulsionSelected;
    }

    public void setRepulsion(double repulsion) {
        this.repulsionSelected = repulsion;
    }
}