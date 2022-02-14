import org.apache.commons.math3.analysis.MultivariateFunction
import java.lang.IllegalArgumentException

class AckleyFunction : MultivariateFunction {
    override fun value(point: DoubleArray): Double {
        require(point.size == 2) { "Dimension must be 2!" }
        return -20 *
                Math.exp(-0.2 * Math.sqrt(0.5 * (point[0] * point[0] + point[1] * point[1]))) -
                Math.exp(0.5 * (Math.cos(2 * Math.PI * point[0]) + Math.cos(2 * Math.PI * point[1]))) +
                Math.exp(1.0) + 20
    }
}