import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.optim.InitialGuess
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.MaxIter
import org.apache.commons.math3.optim.SimpleBounds
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer
import kotlin.jvm.JvmStatic
import kotlin.math.abs

object Main {
    fun calculate(f: MultivariateFunction?, lb: Double, hb: Double) {
        val epsilon = 1e-10
        var counter = 0
        val lowb = lb.toInt()
        val higb = hb.toInt()
        val op = BOBYQAOptimizerKotlin(6)
        for (i in lowb..higb) {
            for (j in lowb..higb) {
                val p = op.optimize(
                    SimpleBounds(doubleArrayOf(lb, lb), doubleArrayOf(hb, hb)),
                    ObjectiveFunction(f),
                    GoalType.MINIMIZE,
                    InitialGuess(doubleArrayOf(i.toDouble(), j.toDouble())),
                    MaxEval(100000),
                    MaxIter(100000)
                )
                val point = p.key
                val value = p.value
                if (abs(0 - value) < epsilon) {
                    counter++
                    print("Optimum point -> ")
                    for (k in point.indices) {
                        print("x_" + (k + 1) + " = " + point[k] + ", ")
                    }
                    print("  Starting point -> ($i, $j)")
                    print("  Optimum value -> $value\n")
                }
            }
        }
        println("Out of " + (higb - lowb + 1) * (higb - lowb + 1) + " starting points found optimum for " + counter)
    }



    @JvmStatic
    fun main(args: Array<String>) {
        //calculate(AckleyFunction(), -5.0, 5.0)
        //calculate(BealeFunction(), -4.5, 4.5)

        val startTime1 = System.nanoTime();
        val op1 = BOBYQAOptimizer(6)
        val p1 = op1.optimize(
            SimpleBounds(doubleArrayOf(-5.0, -5.0), doubleArrayOf(5.0, 5.0)),
            ObjectiveFunction(AckleyFunction()),
            GoalType.MINIMIZE,
            InitialGuess(doubleArrayOf(0.0, 0.0)),
            MaxEval(100000),
            MaxIter(100000)
        )
        val endTime1 = System.nanoTime();


        val startTime2 = System.nanoTime();
        val op2 = BOBYQAOptimizerKotlin(6)
        val p2 = op2.optimize(
            SimpleBounds(doubleArrayOf(-5.0, -5.0), doubleArrayOf(5.0, 5.0)),
            ObjectiveFunction(AckleyFunction()),
            GoalType.MINIMIZE,
            InitialGuess(doubleArrayOf(0.0, 0.0)),
            MaxEval(100000),
            MaxIter(100000)
        )
        val endTime2 = System.nanoTime();

        println("Java klasa = " + (endTime1 - startTime1))
        println("Kotlin klasa = " + (endTime2 - startTime2))


    }
}