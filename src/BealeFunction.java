import org.apache.commons.math3.analysis.MultivariateFunction;

public class BealeFunction implements MultivariateFunction {
    @Override
    public double value(double[] point) {

        if (point.length != 2) {
            throw new IllegalArgumentException("Dimension must be 2!");
        }

        return Math.pow((1.5 - point[0] + point[0] * point[1]), 2) +
                Math.pow((2.25 - point[0] + point[0] * point[1] * point[1]), 2) +
                Math.pow((2.625 - point[0] + point[0] * point[1] * point[1] * point[1]), 2);
    }
}
