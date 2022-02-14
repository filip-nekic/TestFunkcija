import kotlin.jvm.JvmOverloads
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.util.FastMath
import org.apache.commons.math3.exception.MathIllegalStateException
import org.apache.commons.math3.exception.util.LocalizedFormats
import org.apache.commons.math3.linear.RealVector
import org.apache.commons.math3.exception.NumberIsTooSmallException
import org.apache.commons.math3.exception.OutOfRangeException
import org.apache.commons.math3.optim.PointValuePair
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer
import java.lang.RuntimeException
import java.lang.StackTraceElement

/**
 * Powell's BOBYQA algorithm. This implementation is translated and
 * adapted from the Fortran version available
 * [here](http://plato.asu.edu/ftp/other_software/bobyqa.zip).
 * See [
 * this paper](http://www.optimization-online.org/DB_HTML/2010/05/2616.html) for an introduction.
 * <br></br>
 * BOBYQA is particularly well suited for high dimensional problems
 * where derivatives are not available. In most cases it outperforms the
 * [PowellOptimizer] significantly. Stochastic algorithms like
 * [CMAESOptimizer] succeed more often than BOBYQA, but are more
 * expensive. BOBYQA could also be considered as a replacement of any
 * derivative-based optimizer when the derivatives are approximated by
 * finite differences.
 *
 * @since 3.0
 */
class BOBYQAOptimizerKotlin
/**
 * @param numberOfInterpolationPoints Number of interpolation conditions.
 * For a problem of dimension `n`, its value must be in the interval
 * `[n+2, (n+1)(n+2)/2]`.
 * Choices that exceed `2n+1` are not recommended.
 */ @JvmOverloads constructor(
    /**
     * numberOfInterpolationPoints XXX
     */
    private val numberOfInterpolationPoints: Int,
    /**
     * initialTrustRegionRadius XXX
     */
    private var initialTrustRegionRadius: Double =
        DEFAULT_INITIAL_RADIUS,
    /**
     * stoppingTrustRegionRadius XXX
     */
    private val stoppingTrustRegionRadius: Double =
        DEFAULT_STOPPING_RADIUS
) : MultivariateOptimizer(null) {
    /** Goal type (minimize or maximize).  */
    private var isMinimize = false

    /**
     * Current best values for the variables to be optimized.
     * The vector will be changed in-place to contain the values of the least
     * calculated objective function values.
     */
    private var currentBest: ArrayRealVector? = null

    /** Differences between the upper and lower bounds.  */
    private lateinit var boundDifference: DoubleArray

    /**
     * Index of the interpolation point at the trust region center.
     */
    private var trustRegionCenterInterpolationPointIndex = 0

    /**
     * Last *n* columns of matrix H (where *n* is the dimension
     * of the problem).
     * XXX "bmat" in the original code.
     */
    private var bMatrix: Array2DRowRealMatrix? = null

    /**
     * Factorization of the leading *npt* square submatrix of H, this
     * factorization being Z Z<sup>T</sup>, which provides both the correct
     * rank and positive semi-definiteness.
     * XXX "zmat" in the original code.
     */
    private var zMatrix: Array2DRowRealMatrix? = null

    /**
     * Coordinates of the interpolation points relative to [.originShift].
     * XXX "xpt" in the original code.
     */
    private var interpolationPoints: Array2DRowRealMatrix? = null

    /**
     * Shift of origin that should reduce the contributions from rounding
     * errors to values of the model and Lagrange functions.
     * XXX "xbase" in the original code.
     */
    private var originShift: ArrayRealVector? = null

    /**
     * Values of the objective function at the interpolation points.
     * XXX "fval" in the original code.
     */
    private var fAtInterpolationPoints: ArrayRealVector? = null

    /**
     * Displacement from [.originShift] of the trust region center.
     * XXX "xopt" in the original code.
     */
    private var trustRegionCenterOffset: ArrayRealVector? = null

    /**
     * Gradient of the quadratic model at [.originShift] +
     * [.trustRegionCenterOffset].
     * XXX "gopt" in the original code.
     */
    private var gradientAtTrustRegionCenter: ArrayRealVector? = null

    /**
     * Differences [.getLowerBound] - [.originShift].
     * All the components of every [.trustRegionCenterOffset] are going
     * to satisfy the bounds<br></br>
     * [lowerBound][.getLowerBound]<sub>i</sub>
     * [.trustRegionCenterOffset]<sub>i</sub>,<br></br>
     * with appropriate equalities when [.trustRegionCenterOffset] is
     * on a constraint boundary.
     * XXX "sl" in the original code.
     */
    private var lowerDifference: ArrayRealVector? = null

    /**
     * Differences [.getUpperBound] - [.originShift]
     * All the components of every [.trustRegionCenterOffset] are going
     * to satisfy the bounds<br></br>
     * [.trustRegionCenterOffset]<sub>i</sub>
     * [upperBound][.getUpperBound]<sub>i</sub>,<br></br>
     * with appropriate equalities when [.trustRegionCenterOffset] is
     * on a constraint boundary.
     * XXX "su" in the original code.
     */
    private var upperDifference: ArrayRealVector? = null

    /**
     * Parameters of the implicit second derivatives of the quadratic model.
     * XXX "pq" in the original code.
     */
    private var modelSecondDerivativesParameters: ArrayRealVector? = null

    /**
     * Point chosen by function [trsbox][.trsbox]
     * or [altmov][.altmov].
     * Usually [.originShift] + [.newPoint] is the vector of
     * variables for the next evaluation of the objective function.
     * It also satisfies the constraints indicated in [.lowerDifference]
     * and [.upperDifference].
     * XXX "xnew" in the original code.
     */
    private var newPoint: ArrayRealVector? = null

    /**
     * Alternative to [.newPoint], chosen by
     * [altmov][.altmov].
     * It may replace [.newPoint] in order to increase the denominator
     * in the [updating procedure][.update].
     * XXX "xalt" in the original code.
     */
    private var alternativeNewPoint: ArrayRealVector? = null

    /**
     * Trial step from [.trustRegionCenterOffset] which is usually
     * [.newPoint] - [.trustRegionCenterOffset].
     * XXX "d__" in the original code.
     */
    private var trialStepPoint: ArrayRealVector? = null

    /**
     * Values of the Lagrange functions at a new point.
     * XXX "vlag" in the original code.
     */
    private var lagrangeValuesAtNewPoint: ArrayRealVector? = null

    /**
     * Explicit second derivatives of the quadratic model.
     * XXX "hq" in the original code.
     */
    private var modelSecondDerivativesValues: ArrayRealVector? = null
    /**
     * @param numberOfInterpolationPoints Number of interpolation conditions.
     * For a problem of dimension `n`, its value must be in the interval
     * `[n+2, (n+1)(n+2)/2]`.
     * Choices that exceed `2n+1` are not recommended.
     * @param initialTrustRegionRadius Initial trust region radius.
     * @param stoppingTrustRegionRadius Stopping trust region radius.
     */
    /** {@inheritDoc}  */
    override fun doOptimize(): PointValuePair {
        val lowerBound = lowerBound
        val upperBound = upperBound

        // Validity checks.
        setup(lowerBound, upperBound)

        isMinimize = goalType == GoalType.MINIMIZE
        currentBest = ArrayRealVector(startPoint)

        val value = bobyqa(lowerBound, upperBound)

        return PointValuePair(
            currentBest!!.dataRef,
            if (isMinimize) value else -value
        )
    }

    /**
     * This subroutine seeks the least value of a function of many variables,
     * by applying a trust region method that forms quadratic models by
     * interpolation. There is usually some freedom in the interpolation
     * conditions, which is taken up by minimizing the Frobenius norm of
     * the change to the second derivative of the model, beginning with the
     * zero matrix. The values of the variables are constrained by upper and
     * lower bounds. The arguments of the subroutine are as follows.
     *
     * N must be set to the number of variables and must be at least two.
     * NPT is the number of interpolation conditions. Its value must be in
     * the interval [N+2,(N+1)(N+2)/2]. Choices that exceed 2*N+1 are not
     * recommended.
     * Initial values of the variables must be set in X(1),X(2),...,X(N). They
     * will be changed to the values that give the least calculated F.
     * For I=1,2,...,N, XL(I) and XU(I) must provide the lower and upper
     * bounds, respectively, on X(I). The construction of quadratic models
     * requires XL(I) to be strictly less than XU(I) for each I. Further,
     * the contribution to a model from changes to the I-th variable is
     * damaged severely by rounding errors if XU(I)-XL(I) is too small.
     * RHOBEG and RHOEND must be set to the initial and final values of a trust
     * region radius, so both must be positive with RHOEND no greater than
     * RHOBEG. Typically, RHOBEG should be about one tenth of the greatest
     * expected change to a variable, while RHOEND should indicate the
     * accuracy that is required in the final values of the variables. An
     * error return occurs if any of the differences XU(I)-XL(I), I=1,...,N,
     * is less than 2*RHOBEG.
     * MAXFUN must be set to an upper bound on the number of calls of CALFUN.
     * The array W will be used for working space. Its length must be at least
     * (NPT+5)*(NPT+N)+3*N*(N+5)/2.
     *
     * @param lowerBound Lower bounds.
     * @param upperBound Upper bounds.
     * @return the value of the objective at the optimum.
     */
    private fun bobyqa(
        lowerBound: DoubleArray,
        upperBound: DoubleArray
    ): Double {
        printMethod() // XXX
        val n = currentBest!!.dimension

        // Return if there is insufficient space between the bounds. Modify the
        // initial X if necessary in order to avoid conflicts between the bounds
        // and the construction of the first quadratic model. The lower and upper
        // bounds on moves from the updated X are set now, in the ISL and ISU
        // partitions of W, in order to provide useful and exact information about
        // components of X that become within distance RHOBEG from their bounds.
        for (j in 0 until n) {
            val boundDiff = boundDifference[j]
            lowerDifference!!.setEntry(j, lowerBound[j] - currentBest!!.getEntry(j))
            upperDifference!!.setEntry(j, upperBound[j] - currentBest!!.getEntry(j))
            if (lowerDifference!!.getEntry(j) >= -initialTrustRegionRadius) {
                if (lowerDifference!!.getEntry(j) >= ZERO) {
                    currentBest!!.setEntry(j, lowerBound[j])
                    lowerDifference!!.setEntry(j, ZERO)
                    upperDifference!!.setEntry(j, boundDiff)
                } else {
                    currentBest!!.setEntry(j, lowerBound[j] + initialTrustRegionRadius)
                    lowerDifference!!.setEntry(j, -initialTrustRegionRadius)
                    // Computing MAX
                    val deltaOne = upperBound[j] - currentBest!!.getEntry(j)
                    upperDifference!!.setEntry(j, FastMath.max(deltaOne, initialTrustRegionRadius))
                }
            } else if (upperDifference!!.getEntry(j) <= initialTrustRegionRadius) {
                if (upperDifference!!.getEntry(j) <= ZERO) {
                    currentBest!!.setEntry(j, upperBound[j])
                    lowerDifference!!.setEntry(j, -boundDiff)
                    upperDifference!!.setEntry(j, ZERO)
                } else {
                    currentBest!!.setEntry(j, upperBound[j] - initialTrustRegionRadius)
                    // Computing MIN
                    val deltaOne = lowerBound[j] - currentBest!!.getEntry(j)
                    val deltaTwo = -initialTrustRegionRadius
                    lowerDifference!!.setEntry(j, FastMath.min(deltaOne, deltaTwo))
                    upperDifference!!.setEntry(j, initialTrustRegionRadius)
                }
            }
        }

        // Make the call of BOBYQB.
        return bobyqb(lowerBound, upperBound)
    } // bobyqa
    // ----------------------------------------------------------------------------------------
    /**
     * The arguments N, NPT, X, XL, XU, RHOBEG, RHOEND, IPRINT and MAXFUN
     * are identical to the corresponding arguments in SUBROUTINE BOBYQA.
     * XBASE holds a shift of origin that should reduce the contributions
     * from rounding errors to values of the model and Lagrange functions.
     * XPT is a two-dimensional array that holds the coordinates of the
     * interpolation points relative to XBASE.
     * FVAL holds the values of F at the interpolation points.
     * XOPT is set to the displacement from XBASE of the trust region centre.
     * GOPT holds the gradient of the quadratic model at XBASE+XOPT.
     * HQ holds the explicit second derivatives of the quadratic model.
     * PQ contains the parameters of the implicit second derivatives of the
     * quadratic model.
     * BMAT holds the last N columns of H.
     * ZMAT holds the factorization of the leading NPT by NPT submatrix of H,
     * this factorization being ZMAT times ZMAT^T, which provides both the
     * correct rank and positive semi-definiteness.
     * NDIM is the first dimension of BMAT and has the value NPT+N.
     * SL and SU hold the differences XL-XBASE and XU-XBASE, respectively.
     * All the components of every XOPT are going to satisfy the bounds
     * SL(I) .LEQ. XOPT(I) .LEQ. SU(I), with appropriate equalities when
     * XOPT is on a constraint boundary.
     * XNEW is chosen by SUBROUTINE TRSBOX or ALTMOV. Usually XBASE+XNEW is the
     * vector of variables for the next call of CALFUN. XNEW also satisfies
     * the SL and SU constraints in the way that has just been mentioned.
     * XALT is an alternative to XNEW, chosen by ALTMOV, that may replace XNEW
     * in order to increase the denominator in the updating of UPDATE.
     * D is reserved for a trial step from XOPT, which is usually XNEW-XOPT.
     * VLAG contains the values of the Lagrange functions at a new point X.
     * They are part of a product that requires VLAG to be of length NDIM.
     * W is a one-dimensional array that is used for working space. Its length
     * must be at least 3*NDIM = 3*(NPT+N).
     *
     * @param lowerBound Lower bounds.
     * @param upperBound Upper bounds.
     * @return the value of the objective at the optimum.
     */
    private fun bobyqb(
        lowerBound: DoubleArray,
        upperBound: DoubleArray
    ): Double {
        printMethod() // XXX

        val n = currentBest!!.dimension
        val npt = numberOfInterpolationPoints
        val np = n + 1
        val nptm = npt - np
        val nh = n * np / 2

        val work1 = ArrayRealVector(n)
        val work2 = ArrayRealVector(npt)
        val work3 = ArrayRealVector(npt)

        var cauchy = Double.NaN
        var alpha = Double.NaN
        var dsq = Double.NaN
        var crvmin = Double.NaN

        // Set some constants.
        // Parameter adjustments

        // Function Body

        // The call of PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
        // BMAT and ZMAT for the first iteration, with the corresponding values of
        // of NF and KOPT, which are the number of calls of CALFUN so far and the
        // index of the interpolation point at the trust region centre. Then the
        // initial XOPT is set too. The branch to label 720 occurs if MAXFUN is
        // less than NPT. GOPT will be updated if KOPT is different from KBASE.
        trustRegionCenterInterpolationPointIndex = 0
        prelim(lowerBound, upperBound)
        var xoptsq = ZERO
        for (i in 0 until n) {
            trustRegionCenterOffset!!.setEntry(i,
                interpolationPoints!!.getEntry(trustRegionCenterInterpolationPointIndex, i)
            )
            // Computing 2nd power
            val deltaOne = trustRegionCenterOffset!!.getEntry(i)
            xoptsq += deltaOne * deltaOne
        }
        var fsave = fAtInterpolationPoints!!.getEntry(0)
        val kbase = 0

        // Complete the settings that are required for the iterative procedure.
        var ntrits = 0
        var itest = 0
        var knew = 0
        var nfsav = getEvaluations()
        var rho = initialTrustRegionRadius
        var delta = rho
        var diffa = ZERO
        var diffb = ZERO
        var diffc = ZERO
        var f = ZERO
        var beta = ZERO
        var adelt = ZERO
        var denom = ZERO
        var ratio = ZERO
        var dnorm = ZERO
        var scaden = ZERO
        var biglsq = ZERO
        var distsq = ZERO

        // Update GOPT if necessary before the first iteration and after each
        // call of RESCUE that makes a call of CALFUN.
        var state = 20
        while (true) {
            if(state == 20)  {
                printState(20) // XXX
                if (trustRegionCenterInterpolationPointIndex != kbase) {
                    var ih = 0
                    for (j in 0 until n) {
                        for (i in 0..j) {
                            if (i < j) {
                                gradientAtTrustRegionCenter!!.setEntry(j, gradientAtTrustRegionCenter!!.getEntry(j) + modelSecondDerivativesValues!!.getEntry(ih) * trustRegionCenterOffset!!.getEntry(i))
                            }
                            gradientAtTrustRegionCenter!!.setEntry(i, gradientAtTrustRegionCenter!!.getEntry(i) + modelSecondDerivativesValues!!.getEntry(ih) * trustRegionCenterOffset!!.getEntry(j))
                            ih++
                        }
                    }
                    if (getEvaluations() > npt) {
                        for (k in 0 until npt) {
                            var temp = ZERO
                            for (j in 0 until n) {
                                temp += interpolationPoints!!.getEntry(k, j) * trustRegionCenterOffset!!.getEntry(j)
                            }
                            temp *= modelSecondDerivativesParameters!!.getEntry(k)
                            for (i in 0 until n) {
                                gradientAtTrustRegionCenter!!.setEntry(i, gradientAtTrustRegionCenter!!.getEntry(i) + temp * interpolationPoints!!.getEntry(k, i))
                            }
                        }
                        // throw new PathIsExploredException(); // XXX
                    }
                }
                state = 60
            }
            if (state == 60) {
                printState(60)
                val gnew = ArrayRealVector(n)
                val xbdi = ArrayRealVector(n)
                val s = ArrayRealVector(n)
                val hs = ArrayRealVector(n)
                val hred = ArrayRealVector(n)

                val dsqCrvmin = trsbox(
                    delta, gnew, xbdi, s,
                    hs, hred
                )

                dsq = dsqCrvmin[0]
                crvmin = dsqCrvmin[1]

                var deltaOne = delta
                val deltaTwo = FastMath.sqrt(dsq)
                dnorm = FastMath.min(deltaOne, deltaTwo)
                if (dnorm < HALF * rho) {
                    ntrits = -1
                    deltaOne = TEN * rho
                    distsq = deltaOne * deltaOne
                    if (getEvaluations() <= nfsav + 2) {
                        state = 650
                        continue
                    }

                    deltaOne = FastMath.max(diffa, diffb)
                    val errbig = FastMath.max(deltaOne, diffc)
                    val frhosq = rho * ONE_OVER_EIGHT * rho
                    if (crvmin > ZERO &&
                        errbig > frhosq * crvmin
                    ) {
                        state = 650
                        continue
                    }
                    val bdtol = errbig / rho
                    for (j in 0 until n) {
                        var bdtest = bdtol
                        if (newPoint!!.getEntry(j) == lowerDifference!!.getEntry(j)) {
                            bdtest = work1.getEntry(j)
                        }
                        if (newPoint!!.getEntry(j) == upperDifference!!.getEntry(j)) {
                            bdtest = -work1.getEntry(j)
                        }
                        if (bdtest < bdtol) {
                            var curv = modelSecondDerivativesValues!!.getEntry((j + j * j) / 2)
                            for (k in 0 until npt) {
                                val d1 = interpolationPoints!!.getEntry(k, j)
                                curv += modelSecondDerivativesParameters!!.getEntry(k) * (d1 * d1)
                            }
                            bdtest += HALF * curv * rho
                            if (bdtest < bdtol) {
                                state = 650
                                continue
                            }
                        }
                    }
                    state = 680
                    continue
                }
                ++ntrits
                state = 90
            }
            if (state == 90) {
                printState(90)
                if (dsq <= xoptsq * ONE_OVER_A_THOUSAND) {
                    val fracsq = xoptsq * ONE_OVER_FOUR
                    var sumpq = ZERO
                    for (k in 0 until npt) {
                        sumpq += modelSecondDerivativesParameters!!.getEntry(k)
                        var sum = -HALF * xoptsq
                        for (i in 0 until n) {
                            sum += interpolationPoints!!.getEntry(k, i) * trustRegionCenterOffset!!.getEntry(i)
                        }
                        work2.setEntry(k, sum)
                        val temp = fracsq - HALF * sum
                        for (i in 0 until n) {
                            work1.setEntry(i, bMatrix!!.getEntry(k, i))
                            lagrangeValuesAtNewPoint!!.setEntry(i,
                                sum * interpolationPoints!!.getEntry(k, i) + temp * trustRegionCenterOffset!!.getEntry(i))
                            val ip = npt + i
                            for (j in 0..i) {
                                bMatrix!!.setEntry(
                                    ip, j,
                                    bMatrix!!.getEntry(ip, j)
                                            + work1.getEntry(i) * lagrangeValuesAtNewPoint!!.getEntry(j) + lagrangeValuesAtNewPoint!!.getEntry(
                                        i
                                    ) * work1.getEntry(j)
                                )
                            }
                        }
                    }
                    for (m in 0 until nptm) {
                        var sumz = ZERO
                        var sumw = ZERO
                        for (k in 0 until npt) {
                            sumz += zMatrix!!.getEntry(k, m)
                            lagrangeValuesAtNewPoint!!.setEntry(k, work2.getEntry(k) * zMatrix!!.getEntry(k, m))
                            sumw += lagrangeValuesAtNewPoint!!.getEntry(k)
                        }
                        for (j in 0 until n) {
                            var sum = (fracsq * sumz - HALF * sumw) * trustRegionCenterOffset!!.getEntry(j)
                            for (k in 0 until npt) {
                                sum += lagrangeValuesAtNewPoint!!.getEntry(k) * interpolationPoints!!.getEntry(
                                    k,
                                    j
                                )
                            }
                            work1.setEntry(j, sum)
                            for (k in 0 until npt) {
                                bMatrix!!.setEntry(
                                    k, j, bMatrix!!.getEntry(k, j)
                                            + sum * zMatrix!!.getEntry(k, m)
                                )
                            }
                        }
                        for (i in 0 until n) {
                            val ip = i + npt
                            val temp = work1.getEntry(i)
                            for (j in 0..i) {
                                bMatrix!!.setEntry(
                                    ip, j, bMatrix!!.getEntry(ip, j)
                                            + temp * work1.getEntry(j)
                                )
                            }
                        }
                    }

                    var ih = 0
                    for (j in 0 until n) {
                        work1.setEntry(j, -HALF * sumpq * trustRegionCenterOffset!!.getEntry(j))
                        for (k in 0 until npt) {
                            work1.setEntry(j, work1.getEntry(j) + modelSecondDerivativesParameters!!.getEntry(k) * interpolationPoints!!.getEntry(k, j))
                            interpolationPoints!!.setEntry(k, j, interpolationPoints!!.getEntry(k, j) - trustRegionCenterOffset!!.getEntry(j))
                        }
                        for (i in 0..j) {
                            modelSecondDerivativesValues!!.setEntry(
                                ih,
                                modelSecondDerivativesValues!!.getEntry(ih)
                                        + work1.getEntry(i) * trustRegionCenterOffset!!.getEntry(j) + trustRegionCenterOffset!!.getEntry(
                                    i
                                ) * work1.getEntry(j)
                            )
                            bMatrix!!.setEntry(npt + i, j, bMatrix!!.getEntry(npt + j, i))
                            ih++
                        }
                    }
                    for (i in 0 until n) {
                        originShift!!.setEntry(i, originShift!!.getEntry(i) + trustRegionCenterOffset!!.getEntry(i))
                        newPoint!!.setEntry(i, newPoint!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i))
                        lowerDifference!!.setEntry(
                            i,
                            lowerDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)
                        )
                        upperDifference!!.setEntry(
                            i,
                            upperDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)
                        )
                        trustRegionCenterOffset!!.setEntry(i, ZERO)
                    }
                    xoptsq = ZERO
                }
                if (ntrits == 0) {
                    state = 210
                    continue
                }
                state = 230
            }
            if (state == 210) {
                printState(210) // XXX
                // Pick two alternative vectors of variables, relative to XBASE, that
                // are suitable as new positions of the KNEW-th interpolation point.
                // Firstly, XNEW is set to the point on a line through XOPT and another
                // interpolation point that minimizes the predicted value of the next
                // denominator, subject to ||XNEW - XOPT|| .LEQ. ADELT and to the SL
                // and SU bounds. Secondly, XALT is set to the best feasible point on
                // a constrained version of the Cauchy step of the KNEW-th Lagrange
                // function, the corresponding value of the square of this function
                // being returned in CAUCHY. The choice between these alternatives is
                // going to be made when the denominator is calculated.
                val alphaCauchy = altmov(knew, adelt)
                alpha = alphaCauchy[0]
                cauchy = alphaCauchy[1]
                for (i in 0 until n) {
                    trialStepPoint!!.setEntry(i, newPoint!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i))
                }
                state = 230
            }
            if (state == 230)  {
                printState(230)
                for (k in 0 until npt) {
                    var suma = ZERO
                    var sumb = ZERO
                    var sum = ZERO
                    for (j in 0 until n) {
                        suma += interpolationPoints!!.getEntry(k, j) * trialStepPoint!!.getEntry(j)
                        sumb += interpolationPoints!!.getEntry(k, j) * trustRegionCenterOffset!!.getEntry(j)
                        sum += bMatrix!!.getEntry(k, j) * trialStepPoint!!.getEntry(j)
                    }
                    work3.setEntry(k, suma * (HALF * suma + sumb))
                    lagrangeValuesAtNewPoint!!.setEntry(k, sum)
                    work2.setEntry(k, suma)
                }
                beta = ZERO
                for (m in 0 until nptm) {
                    var sum = ZERO
                    for (k in 0 until npt) {
                        sum += zMatrix!!.getEntry(k, m) * work3.getEntry(k)
                    }
                    beta -= sum * sum
                    for (k in 0 until npt) {
                        lagrangeValuesAtNewPoint!!.setEntry(
                            k,
                            lagrangeValuesAtNewPoint!!.getEntry(k) + sum * zMatrix!!.getEntry(k, m)
                        )
                    }
                }
                dsq = ZERO
                var bsum = ZERO
                var dx = ZERO
                for (j in 0 until n) {
                    val d1 = trialStepPoint!!.getEntry(j)
                    dsq += d1 * d1
                    var sum = ZERO
                    for (k in 0 until npt) {
                        sum += work3.getEntry(k) * bMatrix!!.getEntry(k, j)
                    }
                    bsum += sum * trialStepPoint!!.getEntry(j)
                    val jp = npt + j
                    for (i in 0 until n) {
                        sum += bMatrix!!.getEntry(jp, i) * trialStepPoint!!.getEntry(i)
                    }
                    lagrangeValuesAtNewPoint!!.setEntry(jp, sum)
                    bsum += sum * trialStepPoint!!.getEntry(j)
                    dx += trialStepPoint!!.getEntry(j) * trustRegionCenterOffset!!.getEntry(j)
                }
                beta = dx * dx + dsq * (xoptsq + dx + dx + HALF * dsq) + beta - bsum
                lagrangeValuesAtNewPoint!!.setEntry(
                    trustRegionCenterInterpolationPointIndex,
                    lagrangeValuesAtNewPoint!!.getEntry(trustRegionCenterInterpolationPointIndex) + ONE
                )

                if (ntrits == 0) {
                    val d1 = lagrangeValuesAtNewPoint!!.getEntry(knew)
                    denom = d1 * d1 + alpha * beta
                    if (denom < cauchy && cauchy > ZERO) {
                        for (i in 0 until n) {
                            newPoint!!.setEntry(i, alternativeNewPoint!!.getEntry(i))
                            trialStepPoint!!.setEntry(
                                i,
                                newPoint!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)
                            )
                        }
                        cauchy = ZERO
                        state = 230
                        continue
                    }
                } else {
                    val delsq = delta * delta
                    scaden = ZERO
                    biglsq = ZERO
                    knew = 0
                    for (k in 0 until npt) {
                        if (k == trustRegionCenterInterpolationPointIndex) {
                            continue
                        }
                        var hdiag = ZERO
                        for (m in 0 until nptm) {
                            val d1 = zMatrix!!.getEntry(k, m)
                            hdiag += d1 * d1
                        }
                        val d2 = lagrangeValuesAtNewPoint!!.getEntry(k)
                        val den = beta * hdiag + d2 * d2
                        distsq = ZERO
                        for (j in 0 until n) {
                            val d3 =
                                interpolationPoints!!.getEntry(k, j) - trustRegionCenterOffset!!.getEntry(j)
                            distsq += d3 * d3
                        }
                        val d4 = distsq / delsq
                        val temp = FastMath.max(ONE, d4 * d4)
                        if (temp * den > scaden) {
                            scaden = temp * den
                            knew = k
                            denom = den
                        }
                        val d5 = lagrangeValuesAtNewPoint!!.getEntry(k)
                        biglsq = FastMath.max(biglsq, temp * (d5 * d5))
                    }
                }
                state = 360
            }
            if (state == 360) {
                printState(360)
                for (i in 0 until n) {
                    val d3 = lowerBound[i]
                    val d4 = originShift!!.getEntry(i) + newPoint!!.getEntry(i)
                    val d1 = FastMath.max(d3, d4)
                    val d2 = upperBound[i]
                    currentBest!!.setEntry(i, FastMath.min(d1, d2))
                    if (newPoint!!.getEntry(i) == lowerDifference!!.getEntry(i)) {
                        currentBest!!.setEntry(i, lowerBound[i])
                    }
                    if (newPoint!!.getEntry(i) == upperDifference!!.getEntry(i)) {
                        currentBest!!.setEntry(i, upperBound[i])
                    }
                }
                f = computeObjectiveValue(currentBest!!.toArray())
                if (!isMinimize) {
                    f = -f
                }
                if (ntrits == -1) {
                    fsave = f
                    state = 720
                    continue
                }
                val fopt = fAtInterpolationPoints!!.getEntry(trustRegionCenterInterpolationPointIndex)
                var vquad = ZERO
                var ih = 0
                for (j in 0 until n) {
                    vquad += trialStepPoint!!.getEntry(j) * gradientAtTrustRegionCenter!!.getEntry(j)
                    for (i in 0..j) {
                        var temp = trialStepPoint!!.getEntry(i) * trialStepPoint!!.getEntry(j)
                        if (i == j) {
                            temp *= HALF
                        }
                        vquad += modelSecondDerivativesValues!!.getEntry(ih) * temp
                        ih++
                    }
                }
                for (k in 0 until npt) {
                    val d1 = work2.getEntry(k)
                    val d2 = d1 * d1
                    vquad += HALF * modelSecondDerivativesParameters!!.getEntry(k) * d2
                }
                val diff = f - fopt - vquad
                diffc = diffb
                diffb = diffa
                diffa = FastMath.abs(diff)
                if (dnorm > rho) {
                    nfsav = getEvaluations()
                }

                if (ntrits > 0) {
                    if (vquad >= ZERO) {
                        throw MathIllegalStateException(LocalizedFormats.TRUST_REGION_STEP_FAILED, vquad)
                    }
                    ratio = (f - fopt) / vquad
                    val hDelta = HALF * delta
                    delta = if (ratio <= ONE_OVER_TEN) {
                        FastMath.min(hDelta, dnorm)
                    } else if (ratio <= .7) {
                        FastMath.max(hDelta, dnorm)
                    } else {
                        FastMath.max(hDelta, 2 * dnorm)
                    }
                    if (delta <= rho * 1.5) {
                        delta = rho
                    }
                    if (f < fopt) {
                        val ksav = knew
                        val densav = denom
                        val delsq = delta * delta
                        scaden = ZERO
                        biglsq = ZERO
                        knew = 0
                        for (k in 0 until npt) {
                            var hdiag = ZERO
                            for (m in 0 until nptm) {
                                val d1 = zMatrix!!.getEntry(k, m)
                                hdiag += d1 * d1
                            }
                            val d1 = lagrangeValuesAtNewPoint!!.getEntry(k)
                            val den = beta * hdiag + d1 * d1
                            distsq = ZERO
                            for (j in 0 until n) {
                                val d2 = interpolationPoints!!.getEntry(k, j) - newPoint!!.getEntry(j)
                                distsq += d2 * d2
                            }
                            val d3 = distsq / delsq
                            val temp = FastMath.max(ONE, d3 * d3)
                            if (temp * den > scaden) {
                                scaden = temp * den
                                knew = k
                                denom = den
                            }
                            val d4 = lagrangeValuesAtNewPoint!!.getEntry(k)
                            val d5 = temp * (d4 * d4)
                            biglsq = FastMath.max(biglsq, d5)
                        }
                        if (scaden <= HALF * biglsq) {
                            knew = ksav
                            denom = densav
                        }
                    }
                }
                update(beta, denom, knew)
                ih = 0
                val pqold = modelSecondDerivativesParameters!!.getEntry(knew)
                modelSecondDerivativesParameters!!.setEntry(knew, ZERO)
                for (i in 0 until n) {
                    val temp = pqold * interpolationPoints!!.getEntry(knew, i)
                    for (j in 0..i) {
                        modelSecondDerivativesValues!!.setEntry(
                            ih,
                            modelSecondDerivativesValues!!.getEntry(ih) + temp * interpolationPoints!!.getEntry(
                                knew,
                                j
                            )
                        )
                        ih++
                    }
                }
                for (m in 0 until nptm) {
                    val temp = diff * zMatrix!!.getEntry(knew, m)
                    for (k in 0 until npt) {
                        modelSecondDerivativesParameters!!.setEntry(
                            k,
                            modelSecondDerivativesParameters!!.getEntry(k) + temp * zMatrix!!.getEntry(k, m)
                        )
                    }
                }
                fAtInterpolationPoints!!.setEntry(knew, f)
                for (i in 0 until n) {
                    interpolationPoints!!.setEntry(knew, i, newPoint!!.getEntry(i))
                    work1.setEntry(i, bMatrix!!.getEntry(knew, i))
                }
                for (k in 0 until npt) {
                    var suma = ZERO
                    for (m in 0 until nptm) {
                        suma += zMatrix!!.getEntry(knew, m) * zMatrix!!.getEntry(k, m)
                    }
                    var sumb = ZERO
                    for (j in 0 until n) {
                        sumb += interpolationPoints!!.getEntry(k, j) * trustRegionCenterOffset!!.getEntry(j)
                    }
                    val temp = suma * sumb
                    for (i in 0 until n) {
                        work1.setEntry(i, work1.getEntry(i) + temp * interpolationPoints!!.getEntry(k, i))
                    }
                }
                for (i in 0 until n) {
                    gradientAtTrustRegionCenter!!.setEntry(
                        i,
                        gradientAtTrustRegionCenter!!.getEntry(i) + diff * work1.getEntry(i)
                    )
                }
                if (f < fopt) {
                    trustRegionCenterInterpolationPointIndex = knew
                    xoptsq = ZERO
                    ih = 0
                    for (j in 0 until n) {
                        trustRegionCenterOffset!!.setEntry(j, newPoint!!.getEntry(j))
                        val d1 = trustRegionCenterOffset!!.getEntry(j)
                        xoptsq += d1 * d1
                        for (i in 0..j) {
                            if (i < j) {
                                gradientAtTrustRegionCenter!!.setEntry(
                                    j,
                                    gradientAtTrustRegionCenter!!.getEntry(j) + modelSecondDerivativesValues!!.getEntry(
                                        ih
                                    ) * trialStepPoint!!.getEntry(i)
                                )
                            }
                            gradientAtTrustRegionCenter!!.setEntry(
                                i,
                                gradientAtTrustRegionCenter!!.getEntry(i) + modelSecondDerivativesValues!!.getEntry(
                                    ih
                                ) * trialStepPoint!!.getEntry(j)
                            )
                            ih++
                        }
                    }
                    for (k in 0 until npt) {
                        var temp = ZERO
                        for (j in 0 until n) {
                            temp += interpolationPoints!!.getEntry(k, j) * trialStepPoint!!.getEntry(j)
                        }
                        temp *= modelSecondDerivativesParameters!!.getEntry(k)
                        for (i in 0 until n) {
                            gradientAtTrustRegionCenter!!.setEntry(
                                i,
                                gradientAtTrustRegionCenter!!.getEntry(i) + temp * interpolationPoints!!.getEntry(
                                    k,
                                    i
                                )
                            )
                        }
                    }
                }
                if (ntrits > 0) {
                    for (k in 0 until npt) {
                        lagrangeValuesAtNewPoint!!.setEntry(
                            k,
                            fAtInterpolationPoints!!.getEntry(k) - fAtInterpolationPoints!!.getEntry(
                                trustRegionCenterInterpolationPointIndex
                            )
                        )
                        work3.setEntry(k, ZERO)
                    }
                    for (j in 0 until nptm) {
                        var sum = ZERO
                        for (k in 0 until npt) {
                            sum += zMatrix!!.getEntry(k, j) * lagrangeValuesAtNewPoint!!.getEntry(k)
                        }
                        for (k in 0 until npt) {
                            work3.setEntry(k, work3.getEntry(k) + sum * zMatrix!!.getEntry(k, j))
                        }
                    }
                    for (k in 0 until npt) {
                        var sum = ZERO
                        for (j in 0 until n) {
                            sum += interpolationPoints!!.getEntry(k, j) * trustRegionCenterOffset!!.getEntry(j)
                        }
                        work2.setEntry(k, work3.getEntry(k))
                        work3.setEntry(k, sum * work3.getEntry(k))
                    }
                    var gqsq = ZERO
                    var gisq = ZERO
                    for (i in 0 until n) {
                        var sum = ZERO
                        for (k in 0 until npt) {
                            sum += bMatrix!!.getEntry(k, i) *
                                    lagrangeValuesAtNewPoint!!.getEntry(k) + interpolationPoints!!.getEntry(
                                k,
                                i
                            ) * work3.getEntry(k)
                        }
                        if (trustRegionCenterOffset!!.getEntry(i) == lowerDifference!!.getEntry(i)) {
                            val d1 = FastMath.min(ZERO, gradientAtTrustRegionCenter!!.getEntry(i))
                            gqsq += d1 * d1
                            val d2 = FastMath.min(ZERO, sum)
                            gisq += d2 * d2
                        } else if (trustRegionCenterOffset!!.getEntry(i) == upperDifference!!.getEntry(i)) {
                            val d1 = FastMath.max(ZERO, gradientAtTrustRegionCenter!!.getEntry(i))
                            gqsq += d1 * d1
                            val d2 = FastMath.max(ZERO, sum)
                            gisq += d2 * d2
                        } else {
                            val d1 = gradientAtTrustRegionCenter!!.getEntry(i)
                            gqsq += d1 * d1
                            gisq += sum * sum
                        }
                        lagrangeValuesAtNewPoint!!.setEntry(npt + i, sum)
                    }
                    ++itest
                    if (gqsq < TEN * gisq) {
                        itest = 0
                    }
                    if (itest >= 3) {
                        val max = FastMath.max(npt, nh)
                        for (i in 0 until max) {
                            if (i < n) {
                                gradientAtTrustRegionCenter!!.setEntry(
                                    i,
                                    lagrangeValuesAtNewPoint!!.getEntry(npt + i)
                                )
                            }
                            if (i < npt) {
                                modelSecondDerivativesParameters!!.setEntry(i, work2.getEntry(i))
                            }
                            if (i < nh) {
                                modelSecondDerivativesValues!!.setEntry(i, ZERO)
                            }
                            itest = 0
                        }
                    }
                }
                if (ntrits == 0) {
                    state = 60
                    continue
                }
                if (f <= fopt + ONE_OVER_TEN * vquad) {
                    state = 60
                    continue
                }
                val d1 = TWO * delta
                val d2 = TEN * rho
                distsq = FastMath.max(d1 * d1, d2 * d2)
                state = 650
            }
            if (state == 650) {
                printState(650)
                knew = -1
                for (k in 0 until npt) {
                    var sum = ZERO
                    for (j in 0 until n) {
                        val d1 = interpolationPoints!!.getEntry(k, j) - trustRegionCenterOffset!!.getEntry(j)
                        sum += d1 * d1
                    }
                    if (sum > distsq) {
                        knew = k
                        distsq = sum
                    }
                }

                if (knew >= 0) {
                    val dist = FastMath.sqrt(distsq)
                    if (ntrits == -1) {
                        delta = FastMath.min(ONE_OVER_TEN * delta, HALF * dist)
                        if (delta <= rho * 1.5) {
                            delta = rho
                        }
                    }
                    ntrits = 0
                    val d1 = FastMath.min(ONE_OVER_TEN * dist, delta)
                    adelt = FastMath.max(d1, rho)
                    dsq = adelt * adelt
                    state = 90
                    continue
                }
                if (ntrits == -1) {
                    state = 680
                    continue
                }
                if (ratio > ZERO) {
                    state = 60
                    continue
                }
                if (FastMath.max(delta, dnorm) > rho) {
                    state = 60
                    continue
                }
                state = 680
            }
            if (state == 680) {
                printState(680)
                if (rho > stoppingTrustRegionRadius) {
                    delta = HALF * rho
                    ratio = rho / stoppingTrustRegionRadius
                    if (ratio <= SIXTEEN) {
                        rho = stoppingTrustRegionRadius
                    } else if (ratio <= TWO_HUNDRED_FIFTY) {
                        rho = FastMath.sqrt(ratio) * stoppingTrustRegionRadius
                    } else {
                        rho *= ONE_OVER_TEN
                    }
                    delta = FastMath.max(delta, rho)
                    ntrits = 0
                    nfsav = getEvaluations()
                    state = 60
                    continue
                }

                if (ntrits == -1) {
                    state = 360
                    continue
                }
                state = 720
            }
            if (state == 720) {
                printState(720)
                if (fAtInterpolationPoints!!.getEntry(trustRegionCenterInterpolationPointIndex) <= fsave) {
                    for (i in 0 until n) {
                        val d3 = lowerBound[i]
                        val d4 = originShift!!.getEntry(i) + trustRegionCenterOffset!!.getEntry(i)
                        val d1 = FastMath.max(d3, d4)
                        val d2 = upperBound[i]
                        currentBest!!.setEntry(i, FastMath.min(d1, d2))
                        if (trustRegionCenterOffset!!.getEntry(i) == lowerDifference!!.getEntry(i)) {
                            currentBest!!.setEntry(i, lowerBound[i])
                        }
                        if (trustRegionCenterOffset!!.getEntry(i) == upperDifference!!.getEntry(i)) {
                            currentBest!!.setEntry(i, upperBound[i])
                        }
                    }
                    f = fAtInterpolationPoints!!.getEntry(trustRegionCenterInterpolationPointIndex)
                }
                return f
            } else {
                throw MathIllegalStateException(LocalizedFormats.SIMPLE_MESSAGE, "bobyqb")
            }
        }
    }
    // bobyqb
    // ----------------------------------------------------------------------------------------
    /**
     * The arguments N, NPT, XPT, XOPT, BMAT, ZMAT, NDIM, SL and SU all have
     * the same meanings as the corresponding arguments of BOBYQB.
     * KOPT is the index of the optimal interpolation point.
     * KNEW is the index of the interpolation point that is going to be moved.
     * ADELT is the current trust region bound.
     * XNEW will be set to a suitable new position for the interpolation point
     * XPT(KNEW,.). Specifically, it satisfies the SL, SU and trust region
     * bounds and it should provide a large denominator in the next call of
     * UPDATE. The step XNEW-XOPT from XOPT is restricted to moves along the
     * straight lines through XOPT and another interpolation point.
     * XALT also provides a large value of the modulus of the KNEW-th Lagrange
     * function subject to the constraints that have been mentioned, its main
     * difference from XNEW being that XALT-XOPT is a constrained version of
     * the Cauchy step within the trust region. An exception is that XALT is
     * not calculated if all components of GLAG (see below) are zero.
     * ALPHA will be set to the KNEW-th diagonal element of the H matrix.
     * CAUCHY will be set to the square of the KNEW-th Lagrange function at
     * the step XALT-XOPT from XOPT for the vector XALT that is returned,
     * except that CAUCHY is set to zero if XALT is not calculated.
     * GLAG is a working space vector of length N for the gradient of the
     * KNEW-th Lagrange function at XOPT.
     * HCOL is a working space vector of length NPT for the second derivative
     * coefficients of the KNEW-th Lagrange function.
     * W is a working space vector of length 2N that is going to hold the
     * constrained Cauchy step from XOPT of the Lagrange function, followed
     * by the downhill version of XALT when the uphill step is calculated.
     *
     * Set the first NPT components of W to the leading elements of the
     * KNEW-th column of the H matrix.
     * @param knew
     * @param adelt
     */
    private fun altmov(
        knew: Int,
        adelt: Double
    ): DoubleArray {
        printMethod() // XXX

        val n = currentBest!!.dimension
        val npt = numberOfInterpolationPoints

        val glag = ArrayRealVector(n)
        val hcol = ArrayRealVector(npt)

        val work1 = ArrayRealVector(n)
        val work2 = ArrayRealVector(n)

        for (k in 0 until npt) {
            hcol.setEntry(k, ZERO)
        }
        val max = npt - n - 1
        for (j in 0 until max) {
            val tmp = zMatrix!!.getEntry(knew, j)
            for (k in 0 until npt) {
                hcol.setEntry(k, hcol.getEntry(k) + tmp * zMatrix!!.getEntry(k, j))
            }
        }
        val alpha = hcol.getEntry(knew)
        val ha = HALF * alpha

        // Calculate the gradient of the KNEW-th Lagrange function at XOPT.
        for (i in 0 until n) {
            glag.setEntry(i, bMatrix!!.getEntry(knew, i))
        }
        for (k in 0 until npt) {
            var tmp = ZERO
            for (j in 0 until n) {
                tmp += interpolationPoints!!.getEntry(k, j) * trustRegionCenterOffset!!.getEntry(j)
            }
            tmp *= hcol.getEntry(k)
            for (i in 0 until n) {
                glag.setEntry(i, glag.getEntry(i) + tmp * interpolationPoints!!.getEntry(k, i))
            }
        }

        // Search for a large denominator along the straight lines through XOPT
        // and another interpolation point. SLBD and SUBD will be lower and upper
        // bounds on the step along each of these lines in turn. PREDSQ will be
        // set to the square of the predicted denominator for each line. PRESAV
        // will be set to the largest admissible value of PREDSQ that occurs.
        var presav = ZERO
        var step = Double.NaN
        var ksav = 0
        var ibdsav = 0
        var stpsav = 0.0
        for (k in 0 until npt) {
            if (k == trustRegionCenterInterpolationPointIndex) {
                continue
            }
            var dderiv = ZERO
            var distsq = ZERO
            for (i in 0 until n) {
                val tmp = interpolationPoints!!.getEntry(k, i) - trustRegionCenterOffset!!.getEntry(i)
                dderiv += glag.getEntry(i) * tmp
                distsq += tmp * tmp
            }
            var subd = adelt / FastMath.sqrt(distsq)
            var slbd = -subd
            var ilbd = 0
            var iubd = 0
            val sumin = FastMath.min(ONE, subd)

            // Revise SLBD and SUBD if necessary because of the bounds in SL and SU.
            for (i in 0 until n) {
                val tmp = interpolationPoints!!.getEntry(k, i) - trustRegionCenterOffset!!.getEntry(i)
                if (tmp > ZERO) {
                    if (slbd * tmp < lowerDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)) {
                        slbd = (lowerDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)) / tmp
                        ilbd = -i - 1
                    }
                    if (subd * tmp > upperDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)) {
                        // Computing MAX
                        subd = FastMath.max(
                            sumin,
                            (upperDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)) / tmp
                        )
                        iubd = i + 1
                    }
                } else if (tmp < ZERO) {
                    if (slbd * tmp > upperDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)) {
                        slbd = (upperDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)) / tmp
                        ilbd = i + 1
                    }
                    if (subd * tmp < lowerDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)) {
                        // Computing MAX
                        subd = FastMath.max(
                            sumin,
                            (lowerDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)) / tmp
                        )
                        iubd = -i - 1
                    }
                }
            }

            // Seek a large modulus of the KNEW-th Lagrange function when the index
            // of the other interpolation point on the line through XOPT is KNEW.
            step = slbd
            var isbd = ilbd
            var vlag = Double.NaN
            if (k == knew) {
                val diff = dderiv - ONE
                vlag = slbd * (dderiv - slbd * diff)
                val d1 = subd * (dderiv - subd * diff)
                if (FastMath.abs(d1) > FastMath.abs(vlag)) {
                    step = subd
                    vlag = d1
                    isbd = iubd
                }
                val d2 = HALF * dderiv
                val d3 = d2 - diff * slbd
                val d4 = d2 - diff * subd
                if (d3 * d4 < ZERO) {
                    val d5 = d2 * d2 / diff
                    if (FastMath.abs(d5) > FastMath.abs(vlag)) {
                        step = d2 / diff
                        vlag = d5
                        isbd = 0
                    }
                }

                // Search along each of the other lines through XOPT and another point.
            } else {
                vlag = slbd * (ONE - slbd)
                val tmp = subd * (ONE - subd)
                if (FastMath.abs(tmp) > FastMath.abs(vlag)) {
                    step = subd
                    vlag = tmp
                    isbd = iubd
                }
                if (subd > HALF && FastMath.abs(vlag) < ONE_OVER_FOUR) {
                    step = HALF
                    vlag = ONE_OVER_FOUR
                    isbd = 0
                }
                vlag *= dderiv
            }

            // Calculate PREDSQ for the current line search and maintain PRESAV.
            val tmp = step * (ONE - step) * distsq
            val predsq = vlag * vlag * (vlag * vlag + ha * tmp * tmp)
            if (predsq > presav) {
                presav = predsq
                ksav = k
                stpsav = step
                ibdsav = isbd
            }
        }

        // Construct XNEW in a way that satisfies the bound constraints exactly.
        for (i in 0 until n) {
            val tmp = trustRegionCenterOffset!!.getEntry(i) + stpsav * (interpolationPoints!!.getEntry(
                ksav,
                i
            ) - trustRegionCenterOffset!!.getEntry(i))
            newPoint!!.setEntry(
                i, FastMath.max(
                    lowerDifference!!.getEntry(i),
                    FastMath.min(upperDifference!!.getEntry(i), tmp)
                )
            )
        }
        if (ibdsav < 0) {
            newPoint!!.setEntry(-ibdsav - 1, lowerDifference!!.getEntry(-ibdsav - 1))
        }
        if (ibdsav > 0) {
            newPoint!!.setEntry(ibdsav - 1, upperDifference!!.getEntry(ibdsav - 1))
        }

        // Prepare for the iterative method that assembles the constrained Cauchy
        // step in W. The sum of squares of the fixed components of W is formed in
        // WFIXSQ, and the free components of W are set to BIGSTP.
        val bigstp = adelt + adelt
        var iflag = 0
        var cauchy = Double.NaN
        var csave = ZERO
        while (true) {
            var wfixsq = ZERO
            var ggfree = ZERO
            for (i in 0 until n) {
                val glagValue = glag.getEntry(i)
                work1.setEntry(i, ZERO)
                if (FastMath.min(
                        trustRegionCenterOffset!!.getEntry(i) - lowerDifference!!.getEntry(i),
                        glagValue
                    ) > ZERO ||
                    FastMath.max(
                        trustRegionCenterOffset!!.getEntry(i) - upperDifference!!.getEntry(i),
                        glagValue
                    ) < ZERO
                ) {
                    work1.setEntry(i, bigstp)
                    // Computing 2nd power
                    ggfree += glagValue * glagValue
                }
            }
            if (ggfree == ZERO) {
                return doubleArrayOf(alpha, ZERO)
            }

            // Investigate whether more components of W can be fixed.
            val tmp1 = adelt * adelt - wfixsq
            if (tmp1 > ZERO) {
                step = FastMath.sqrt(tmp1 / ggfree)
                ggfree = ZERO
                for (i in 0 until n) {
                    if (work1.getEntry(i) == bigstp) {
                        val tmp2 = trustRegionCenterOffset!!.getEntry(i) - step * glag.getEntry(i)
                        if (tmp2 <= lowerDifference!!.getEntry(i)) {
                            work1.setEntry(i, lowerDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i))
                            // Computing 2nd power
                            val d1 = work1.getEntry(i)
                            wfixsq += d1 * d1
                        } else if (tmp2 >= upperDifference!!.getEntry(i)) {
                            work1.setEntry(i, upperDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i))
                            // Computing 2nd power
                            val d1 = work1.getEntry(i)
                            wfixsq += d1 * d1
                        } else {
                            // Computing 2nd power
                            val d1 = glag.getEntry(i)
                            ggfree += d1 * d1
                        }
                    }
                }
            }

            // Set the remaining free components of W and all components of XALT,
            // except that W may be scaled later.
            var gw = ZERO
            for (i in 0 until n) {
                val glagValue = glag.getEntry(i)
                if (work1.getEntry(i) == bigstp) {
                    work1.setEntry(i, -step * glagValue)
                    val min = FastMath.min(
                        upperDifference!!.getEntry(i),
                        trustRegionCenterOffset!!.getEntry(i) + work1.getEntry(i)
                    )
                    alternativeNewPoint!!.setEntry(i, FastMath.max(lowerDifference!!.getEntry(i), min))
                } else if (work1.getEntry(i) == ZERO) {
                    alternativeNewPoint!!.setEntry(i, trustRegionCenterOffset!!.getEntry(i))
                } else if (glagValue > ZERO) {
                    alternativeNewPoint!!.setEntry(i, lowerDifference!!.getEntry(i))
                } else {
                    alternativeNewPoint!!.setEntry(i, upperDifference!!.getEntry(i))
                }
                gw += glagValue * work1.getEntry(i)
            }

            // Set CURV to the curvature of the KNEW-th Lagrange function along W.
            // Scale W by a factor less than one if that can reduce the modulus of
            // the Lagrange function at XOPT+W. Set CAUCHY to the final value of
            // the square of this function.
            var curv = ZERO
            for (k in 0 until npt) {
                var tmp = ZERO
                for (j in 0 until n) {
                    tmp += interpolationPoints!!.getEntry(k, j) * work1.getEntry(j)
                }
                curv += hcol.getEntry(k) * tmp * tmp
            }
            if (iflag == 1) {
                curv = -curv
            }
            cauchy = if (curv > -gw &&
                curv < -gw * (ONE + FastMath.sqrt(TWO))
            ) {
                val scale = -gw / curv
                for (i in 0 until n) {
                    val tmp = trustRegionCenterOffset!!.getEntry(i) + scale * work1.getEntry(i)
                    alternativeNewPoint!!.setEntry(
                        i, FastMath.max(
                            lowerDifference!!.getEntry(i),
                            FastMath.min(upperDifference!!.getEntry(i), tmp)
                        )
                    )
                }
                // Computing 2nd power
                val d1 = HALF * gw * scale
                d1 * d1
            } else {
                // Computing 2nd power
                val d1 = gw + HALF * curv
                d1 * d1
            }

            // If IFLAG is zero, then XALT is calculated as before after reversing
            // the sign of GLAG. Thus two XALT vectors become available. The one that
            // is chosen is the one that gives the larger value of CAUCHY.
            if (iflag == 0) {
                for (i in 0 until n) {
                    glag.setEntry(i, -glag.getEntry(i))
                    work2.setEntry(i, alternativeNewPoint!!.getEntry(i))
                }
                csave = cauchy
                iflag = 1
            } else {
                break
            }
        }
        if (csave > cauchy) {
            for (i in 0 until n) {
                alternativeNewPoint!!.setEntry(i, work2.getEntry(i))
            }
            cauchy = csave
        }
        return doubleArrayOf(alpha, cauchy)
    } // altmov
    // ----------------------------------------------------------------------------------------
    /**
     * SUBROUTINE PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
     * BMAT and ZMAT for the first iteration, and it maintains the values of
     * NF and KOPT. The vector X is also changed by PRELIM.
     *
     * The arguments N, NPT, X, XL, XU, RHOBEG, IPRINT and MAXFUN are the
     * same as the corresponding arguments in SUBROUTINE BOBYQA.
     * The arguments XBASE, XPT, FVAL, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU
     * are the same as the corresponding arguments in BOBYQB, the elements
     * of SL and SU being set in BOBYQA.
     * GOPT is usually the gradient of the quadratic model at XOPT+XBASE, but
     * it is set by PRELIM to the gradient of the quadratic model at XBASE.
     * If XOPT is nonzero, BOBYQB will change it to its usual value later.
     * NF is maintaned as the number of calls of CALFUN so far.
     * KOPT will be such that the least calculated value of F so far is at
     * the point XPT(KOPT,.)+XBASE in the space of the variables.
     *
     * @param lowerBound Lower bounds.
     * @param upperBound Upper bounds.
     */
    private fun prelim(
        lowerBound: DoubleArray,
        upperBound: DoubleArray
    ) {
        printMethod() // XXX

        val n = currentBest!!.dimension
        val npt = numberOfInterpolationPoints
        val ndim = bMatrix!!.rowDimension

        val rhosq = initialTrustRegionRadius * initialTrustRegionRadius
        val recip = 1.0 / rhosq
        val np = n + 1

        // Set XBASE to the initial vector of variables, and set the initial
        // elements of XPT, BMAT, HQ, PQ and ZMAT to zero.
        for (j in 0 until n) {
            originShift!!.setEntry(j, currentBest!!.getEntry(j))
            for (k in 0 until npt) {
                interpolationPoints!!.setEntry(k, j, ZERO)
            }
            for (i in 0 until ndim) {
                bMatrix!!.setEntry(i, j, ZERO)
            }
        }
        val max = n * np / 2
        for (i in 0 until max) {
            modelSecondDerivativesValues!!.setEntry(i, ZERO)
        }
        for (k in 0 until npt) {
            modelSecondDerivativesParameters!!.setEntry(k, ZERO)
            val max = npt - np
            for (j in 0 until max) {
                zMatrix!!.setEntry(k, j, ZERO)
            }
        }

        // Begin the initialization procedure. NF becomes one more than the number
        // of function values so far. The coordinates of the displacement of the
        // next initial interpolation point from XBASE are set in XPT(NF+1,.).
        var ipt = 0
        var jpt = 0
        var fbeg = Double.NaN
        do {
            val nfm = getEvaluations()
            val nfx = nfm - n
            val nfmm = nfm - 1
            val nfxm = nfx - 1
            var stepa = 0.0
            var stepb = 0.0
            if (nfm <= 2 * n) {
                if (nfm >= 1 &&
                    nfm <= n
                ) {
                    stepa = initialTrustRegionRadius
                    if (upperDifference!!.getEntry(nfmm) == ZERO) {
                        stepa = -stepa
                        // throw new PathIsExploredException(); // XXX
                    }
                    interpolationPoints!!.setEntry(nfm, nfmm, stepa)
                } else if (nfm > n) {
                    stepa = interpolationPoints!!.getEntry(nfx, nfxm)
                    stepb = -initialTrustRegionRadius
                    if (lowerDifference!!.getEntry(nfxm) == ZERO) {
                        stepb = FastMath.min(TWO * initialTrustRegionRadius, upperDifference!!.getEntry(nfxm))
                        // throw new PathIsExploredException(); // XXX
                    }
                    if (upperDifference!!.getEntry(nfxm) == ZERO) {
                        stepb = FastMath.max(-TWO * initialTrustRegionRadius, lowerDifference!!.getEntry(nfxm))
                        // throw new PathIsExploredException(); // XXX
                    }
                    interpolationPoints!!.setEntry(nfm, nfxm, stepb)
                }
            } else {
                val tmp1 = (nfm - np) / n
                jpt = nfm - tmp1 * n - n
                ipt = jpt + tmp1
                if (ipt > n) {
                    val tmp2 = jpt
                    jpt = ipt - n
                    ipt = tmp2
                    //                     throw new PathIsExploredException(); // XXX
                }
                val iptMinus1 = ipt - 1
                val jptMinus1 = jpt - 1
                interpolationPoints!!.setEntry(nfm, iptMinus1, interpolationPoints!!.getEntry(ipt, iptMinus1))
                interpolationPoints!!.setEntry(nfm, jptMinus1, interpolationPoints!!.getEntry(jpt, jptMinus1))
            }

            // Calculate the next value of F. The least function value so far and
            // its index are required.
            for (j in 0 until n) {
                currentBest!!.setEntry(
                    j, FastMath.min(
                        FastMath.max(
                            lowerBound[j],
                            originShift!!.getEntry(j) + interpolationPoints!!.getEntry(nfm, j)
                        ),
                        upperBound[j]
                    )
                )
                if (interpolationPoints!!.getEntry(nfm, j) == lowerDifference!!.getEntry(j)) {
                    currentBest!!.setEntry(j, lowerBound[j])
                }
                if (interpolationPoints!!.getEntry(nfm, j) == upperDifference!!.getEntry(j)) {
                    currentBest!!.setEntry(j, upperBound[j])
                }
            }
            val objectiveValue = computeObjectiveValue(currentBest!!.toArray())
            val f = if (isMinimize) objectiveValue else -objectiveValue
            val numEval = getEvaluations() // nfm + 1
            fAtInterpolationPoints!!.setEntry(nfm, f)
            if (numEval == 1) {
                fbeg = f
                trustRegionCenterInterpolationPointIndex = 0
            } else if (f < fAtInterpolationPoints!!.getEntry(trustRegionCenterInterpolationPointIndex)) {
                trustRegionCenterInterpolationPointIndex = nfm
            }

            // Set the nonzero initial elements of BMAT and the quadratic model in the
            // cases when NF is at most 2*N+1. If NF exceeds N+1, then the positions
            // of the NF-th and (NF-N)-th interpolation points may be switched, in
            // order that the function value at the first of them contributes to the
            // off-diagonal second derivative terms of the initial quadratic model.
            if (numEval <= 2 * n + 1) {
                if (numEval >= 2 &&
                    numEval <= n + 1
                ) {
                    gradientAtTrustRegionCenter!!.setEntry(nfmm, (f - fbeg) / stepa)
                    if (npt < numEval + n) {
                        val oneOverStepA = ONE / stepa
                        bMatrix!!.setEntry(0, nfmm, -oneOverStepA)
                        bMatrix!!.setEntry(nfm, nfmm, oneOverStepA)
                        bMatrix!!.setEntry(npt + nfmm, nfmm, -HALF * rhosq)
                        // throw new PathIsExploredException(); // XXX
                    }
                } else if (numEval >= n + 2) {
                    val ih = nfx * (nfx + 1) / 2 - 1
                    val tmp = (f - fbeg) / stepb
                    val diff = stepb - stepa
                    modelSecondDerivativesValues!!.setEntry(
                        ih,
                        TWO * (tmp - gradientAtTrustRegionCenter!!.getEntry(nfxm)) / diff
                    )
                    gradientAtTrustRegionCenter!!.setEntry(
                        nfxm,
                        (gradientAtTrustRegionCenter!!.getEntry(nfxm) * stepb - tmp * stepa) / diff
                    )
                    if (stepa * stepb < ZERO && f < fAtInterpolationPoints!!.getEntry(nfm - n)) {
                        fAtInterpolationPoints!!.setEntry(nfm, fAtInterpolationPoints!!.getEntry(nfm - n))
                        fAtInterpolationPoints!!.setEntry(nfm - n, f)
                        if (trustRegionCenterInterpolationPointIndex == nfm) {
                            trustRegionCenterInterpolationPointIndex = nfm - n
                        }
                        interpolationPoints!!.setEntry(nfm - n, nfxm, stepb)
                        interpolationPoints!!.setEntry(nfm, nfxm, stepa)
                    }
                    bMatrix!!.setEntry(0, nfxm, -(stepa + stepb) / (stepa * stepb))
                    bMatrix!!.setEntry(nfm, nfxm, -HALF / interpolationPoints!!.getEntry(nfm - n, nfxm))
                    bMatrix!!.setEntry(
                        nfm - n, nfxm,
                        -bMatrix!!.getEntry(0, nfxm) - bMatrix!!.getEntry(nfm, nfxm)
                    )
                    zMatrix!!.setEntry(0, nfxm, FastMath.sqrt(TWO) / (stepa * stepb))
                    zMatrix!!.setEntry(nfm, nfxm, FastMath.sqrt(HALF) / rhosq)
                    // zMatrix.setEntry(nfm, nfxm, FastMath.sqrt(HALF) * recip); // XXX "testAckley" and "testDiffPow" fail.
                    zMatrix!!.setEntry(
                        nfm - n, nfxm,
                        -zMatrix!!.getEntry(0, nfxm) - zMatrix!!.getEntry(nfm, nfxm)
                    )
                }

                // Set the off-diagonal second derivatives of the Lagrange functions and
                // the initial quadratic model.
            } else {
                zMatrix!!.setEntry(0, nfxm, recip)
                zMatrix!!.setEntry(nfm, nfxm, recip)
                zMatrix!!.setEntry(ipt, nfxm, -recip)
                zMatrix!!.setEntry(jpt, nfxm, -recip)
                val ih = ipt * (ipt - 1) / 2 + jpt - 1
                val tmp = interpolationPoints!!.getEntry(nfm, ipt - 1) * interpolationPoints!!.getEntry(nfm, jpt - 1)
                modelSecondDerivativesValues!!.setEntry(
                    ih,
                    (fbeg - fAtInterpolationPoints!!.getEntry(ipt) - fAtInterpolationPoints!!.getEntry(jpt) + f) / tmp
                )
                //                 throw new PathIsExploredException(); // XXX
            }
        } while (getEvaluations() < npt)
    } // prelim
    // ----------------------------------------------------------------------------------------
    /**
     * A version of the truncated conjugate gradient is applied. If a line
     * search is restricted by a constraint, then the procedure is restarted,
     * the values of the variables that are at their bounds being fixed. If
     * the trust region boundary is reached, then further changes may be made
     * to D, each one being in the two dimensional space that is spanned
     * by the current D and the gradient of Q at XOPT+D, staying on the trust
     * region boundary. Termination occurs when the reduction in Q seems to
     * be close to the greatest reduction that can be achieved.
     * The arguments N, NPT, XPT, XOPT, GOPT, HQ, PQ, SL and SU have the same
     * meanings as the corresponding arguments of BOBYQB.
     * DELTA is the trust region radius for the present calculation, which
     * seeks a small value of the quadratic model within distance DELTA of
     * XOPT subject to the bounds on the variables.
     * XNEW will be set to a new vector of variables that is approximately
     * the one that minimizes the quadratic model within the trust region
     * subject to the SL and SU constraints on the variables. It satisfies
     * as equations the bounds that become active during the calculation.
     * D is the calculated trial step from XOPT, generated iteratively from an
     * initial value of zero. Thus XNEW is XOPT+D after the final iteration.
     * GNEW holds the gradient of the quadratic model at XOPT+D. It is updated
     * when D is updated.
     * xbdi.get( is a working space vector. For I=1,2,...,N, the element xbdi.get((I) is
     * set to -1.0, 0.0, or 1.0, the value being nonzero if and only if the
     * I-th variable has become fixed at a bound, the bound being SL(I) or
     * SU(I) in the case xbdi.get((I)=-1.0 or xbdi.get((I)=1.0, respectively. This
     * information is accumulated during the construction of XNEW.
     * The arrays S, HS and HRED are also used for working space. They hold the
     * current search direction, and the changes in the gradient of Q along S
     * and the reduced D, respectively, where the reduced D is the same as D,
     * except that the components of the fixed variables are zero.
     * DSQ will be set to the square of the length of XNEW-XOPT.
     * CRVMIN is set to zero if D reaches the trust region boundary. Otherwise
     * it is set to the least curvature of H that occurs in the conjugate
     * gradient searches that are not restricted by any constraints. The
     * value CRVMIN=-1.0D0 is set, however, if all of these searches are
     * constrained.
     * @param delta
     * @param gnew
     * @param xbdi
     * @param s
     * @param hs
     * @param hred
     */
    private fun trsbox(
        delta: Double,
        gnew: ArrayRealVector,
        xbdi: ArrayRealVector,
        s: ArrayRealVector,
        hs: ArrayRealVector,
        hred: ArrayRealVector
    ): DoubleArray {
        printMethod() // XXX
        val n = currentBest!!.dimension
        val npt = numberOfInterpolationPoints
        var dsq = Double.NaN
        var crvmin = Double.NaN

        // Local variables
        var ds: Double
        var iu: Int
        var dhd: Double
        var dhs: Double
        var cth: Double
        var shs: Double
        var sth: Double
        var ssq: Double
        var beta = 0.0
        var sdec: Double
        var blen: Double
        var iact = -1
        var nact = 0
        var angt = 0.0
        var qred: Double
        var isav: Int
        var temp = 0.0
        var xsav = 0.0
        var xsum = 0.0
        var angbd = 0.0
        var dredg = 0.0
        var sredg = 0.0
        var iterc: Int
        var resid = 0.0
        var delsq = 0.0
        var ggsav = 0.0
        var tempa = 0.0
        var tempb = 0.0
        var redmax = 0.0
        var dredsq = 0.0
        var redsav = 0.0
        var gredsq = 0.0
        var rednew = 0.0
        var itcsav = 0
        var rdprev = 0.0
        var rdnext = 0.0
        var stplen = 0.0
        var stepsq = 0.0
        var itermax = 0

        // Set some constants.

        // Function Body

        // The sign of GOPT(I) gives the sign of the change to the I-th variable
        // that will reduce Q from its value at XOPT. Thus xbdi.get((I) shows whether
        // or not to fix the I-th variable at one of its bounds initially, with
        // NACT being set to the number of fixed variables. D and GNEW are also
        // set for the first iteration. DELSQ is the upper bound on the sum of
        // squares of the free variables. QRED is the reduction in Q so far.
        iterc = 0
        nact = 0
        for (i in 0 until n) {
            xbdi.setEntry(i, ZERO)
            if (trustRegionCenterOffset!!.getEntry(i) <= lowerDifference!!.getEntry(i)) {
                if (gradientAtTrustRegionCenter!!.getEntry(i) >= ZERO) {
                    xbdi.setEntry(i, MINUS_ONE)
                }
            } else if (trustRegionCenterOffset!!.getEntry(i) >= upperDifference!!.getEntry(i) &&
                gradientAtTrustRegionCenter!!.getEntry(i) <= ZERO
            ) {
                xbdi.setEntry(i, ONE)
            }
            if (xbdi.getEntry(i) != ZERO) {
                ++nact
            }
            trialStepPoint!!.setEntry(i, ZERO)
            gnew.setEntry(i, gradientAtTrustRegionCenter!!.getEntry(i))
        }
        delsq = delta * delta
        qred = ZERO
        crvmin = MINUS_ONE

        // Set the next search direction of the conjugate gradient method. It is
        // the steepest descent direction initially and when the iterations are
        // restarted because a variable has just been fixed by a bound, and of
        // course the components of the fixed variables are zero. ITERMAX is an
        // upper bound on the indices of the conjugate gradient iterations.
        var state = 20
        while (true) {
            if (state == 20) {
                printState(20) // XXX
                beta = ZERO
                state = 30
            }
            if (state == 30) {
                printState(30)
                stepsq = ZERO
                for (i in 0 until n) {
                    if (xbdi.getEntry(i) != ZERO) {
                        s.setEntry(i, ZERO)
                    } else if (beta == ZERO) {
                        s.setEntry(i, -gnew.getEntry(i))
                    } else {
                        s.setEntry(i, beta * s.getEntry(i) - gnew.getEntry(i))
                    }
                    val d1 = s.getEntry(i)
                    stepsq += d1 * d1
                }
                if (stepsq == ZERO) {
                    state = 190
                    continue
                }
                if (beta == ZERO) {
                    gredsq = stepsq
                    itermax = iterc + n - nact
                }
                if (gredsq * delsq <= qred * 1e-4 * qred) {
                    state = 190
                    continue
                }
                state = 210
                continue
            }
            if (state == 50) {
                printState(50) // XXX
                resid = delsq
                ds = ZERO
                shs = ZERO
                for (i in 0 until n) {
                    if (xbdi.getEntry(i) == ZERO) {
                        // Computing 2nd power
                        val d1 = trialStepPoint!!.getEntry(i)
                        resid -= d1 * d1
                        ds += s.getEntry(i) * trialStepPoint!!.getEntry(i)
                        shs += s.getEntry(i) * hs.getEntry(i)
                    }
                }
                if (resid <= ZERO) {
                    state = 90
                    continue
                }
                temp = FastMath.sqrt(stepsq * resid + ds * ds)
                blen = if (ds < ZERO) {
                    (temp - ds) / stepsq
                } else {
                    resid / (temp + ds)
                }
                stplen = blen
                if (shs > ZERO) {
                    // Computing MIN
                    stplen = FastMath.min(blen, gredsq / shs)
                }

                // Reduce STPLEN if necessary in order to preserve the simple bounds,
                // letting IACT be the index of the new constrained variable.
                iact = -1
                for (i in 0 until n) {
                    if (s.getEntry(i) != ZERO) {
                        xsum = trustRegionCenterOffset!!.getEntry(i) + trialStepPoint!!.getEntry(i)
                        temp = if (s.getEntry(i) > ZERO) {
                            (upperDifference!!.getEntry(i) - xsum) / s.getEntry(i)
                        } else {
                            (lowerDifference!!.getEntry(i) - xsum) / s.getEntry(i)
                        }
                        if (temp < stplen) {
                            stplen = temp
                            iact = i
                        }
                    }
                }

                // Update CRVMIN, GNEW and D. Set SDEC to the decrease that occurs in Q.
                sdec = ZERO
                if (stplen > ZERO) {
                    ++iterc
                    temp = shs / stepsq
                    if (iact == -1 && temp > ZERO) {
                        crvmin = FastMath.min(crvmin, temp)
                        if (crvmin == MINUS_ONE) {
                            crvmin = temp
                        }
                    }
                    ggsav = gredsq
                    gredsq = ZERO
                    for (i in 0 until n) {
                        gnew.setEntry(i, gnew.getEntry(i) + stplen * hs.getEntry(i))
                        if (xbdi.getEntry(i) == ZERO) {
                            // Computing 2nd power
                            val d1 = gnew.getEntry(i)
                            gredsq += d1 * d1
                        }
                        trialStepPoint!!.setEntry(i, trialStepPoint!!.getEntry(i) + stplen * s.getEntry(i))
                    }
                    // Computing MAX
                    val d1 = stplen * (ggsav - HALF * stplen * shs)
                    sdec = FastMath.max(d1, ZERO)
                    qred += sdec
                }

                // Restart the conjugate gradient method if it has hit a new bound.
                if (iact >= 0) {
                    ++nact
                    xbdi.setEntry(iact, ONE)
                    if (s.getEntry(iact) < ZERO) {
                        xbdi.setEntry(iact, MINUS_ONE)
                    }
                    // Computing 2nd power
                    val d1 = trialStepPoint!!.getEntry(iact)
                    delsq -= d1 * d1
                    if (delsq <= ZERO) {
                        state = 190
                        continue
                    }
                    state = 20
                    continue
                }

                // If STPLEN is less than BLEN, then either apply another conjugate
                // gradient iteration or RETURN.
                if (stplen < blen) {
                    if (iterc == itermax) {
                        state = 190
                        continue
                    }
                    if (sdec <= qred * .01) {
                        state = 190
                        continue
                    }
                    beta = gredsq / ggsav
                    state = 30
                    continue
                }
                state = 90
            }
            if (state == 90) {
                printState(90)
                crvmin = ZERO
                state = 100
            }
            if (state == 100) {
                printState(100)
                if (nact >= n - 1) {
                    state = 190
                    continue
                }
                dredsq = ZERO
                dredg = ZERO
                gredsq = ZERO
                for (i in 0 until n) {
                    if (xbdi.getEntry(i) == ZERO) {
                        var d1 = trialStepPoint!!.getEntry(i)
                        dredsq += d1 * d1
                        dredg += trialStepPoint!!.getEntry(i) * gnew.getEntry(i)
                        d1 = gnew.getEntry(i)
                        gredsq += d1 * d1
                        s.setEntry(i, trialStepPoint!!.getEntry(i))
                    } else {
                        s.setEntry(i, ZERO)
                    }
                }
                itcsav = iterc
                state = 210
            }
            if (state == 120) {
                printState(120) // XXX
                ++iterc
                temp = gredsq * dredsq - dredg * dredg
                if (temp <= qred * 1e-4 * qred) {
                    state = 190
                    continue
                }
                temp = FastMath.sqrt(temp)
                for (i in 0 until n) {
                    if (xbdi.getEntry(i) == ZERO) {
                        s.setEntry(i, (dredg * trialStepPoint!!.getEntry(i) - dredsq * gnew.getEntry(i)) / temp)
                    } else {
                        s.setEntry(i, ZERO)
                    }
                }
                sredg = -temp

                // By considering the simple bounds on the variables, calculate an upper
                // bound on the tangent of half the angle of the alternative iteration,
                // namely ANGBD, except that, if already a free variable has reached a
                // bound, there is a branch back to label 100 after fixing that variable.
                angbd = ONE
                iact = -1
                for (i in 0 until n) {
                    if (xbdi.getEntry(i) == ZERO) {
                        tempa =
                            trustRegionCenterOffset!!.getEntry(i) + trialStepPoint!!.getEntry(i) - lowerDifference!!.getEntry(
                                i
                            )
                        tempb =
                            upperDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i) - trialStepPoint!!.getEntry(
                                i
                            )
                        if (tempa <= ZERO) {
                            ++nact
                            xbdi.setEntry(i, MINUS_ONE)
                            state = 100
                            continue
                        } else if (tempb <= ZERO) {
                            ++nact
                            xbdi.setEntry(i, ONE)
                            state = 100
                            continue
                        }
                        // Computing 2nd power
                        var d1 = trialStepPoint!!.getEntry(i)
                        // Computing 2nd power
                        val d2 = s.getEntry(i)
                        ssq = d1 * d1 + d2 * d2
                        // Computing 2nd power
                        d1 = trustRegionCenterOffset!!.getEntry(i) - lowerDifference!!.getEntry(i)
                        temp = ssq - d1 * d1
                        if (temp > ZERO) {
                            temp = FastMath.sqrt(temp) - s.getEntry(i)
                            if (angbd * temp > tempa) {
                                angbd = tempa / temp
                                iact = i
                                xsav = MINUS_ONE
                            }
                        }
                        // Computing 2nd power
                        d1 = upperDifference!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i)
                        temp = ssq - d1 * d1
                        if (temp > ZERO) {
                            temp = FastMath.sqrt(temp) + s.getEntry(i)
                            if (angbd * temp > tempb) {
                                angbd = tempb / temp
                                iact = i
                                xsav = ONE
                            }
                        }
                    }
                }

                // Calculate HHD and some curvatures for the alternative iteration.
                state = 210
            }
            if (state == 150) {
                printState(150) // XXX
                shs = ZERO
                dhs = ZERO
                dhd = ZERO
                for (i in 0 until n) {
                    if (xbdi.getEntry(i) == ZERO) {
                        shs += s.getEntry(i) * hs.getEntry(i)
                        dhs += trialStepPoint!!.getEntry(i) * hs.getEntry(i)
                        dhd += trialStepPoint!!.getEntry(i) * hred.getEntry(i)
                    }
                }

                // Seek the greatest reduction in Q for a range of equally spaced values
                // of ANGT in [0,ANGBD], where ANGT is the tangent of half the angle of
                // the alternative iteration.
                redmax = ZERO
                isav = -1
                redsav = ZERO
                iu = (angbd * 17.0 + 3.1).toInt()
                for (i in 0 until iu) {
                    angt = angbd * i / iu
                    sth = (angt + angt) / (ONE + angt * angt)
                    temp = shs + angt * (angt * dhd - dhs - dhs)
                    rednew = sth * (angt * dredg - sredg - HALF * sth * temp)
                    if (rednew > redmax) {
                        redmax = rednew
                        isav = i
                        rdprev = redsav
                    } else if (i == isav + 1) {
                        rdnext = rednew
                    }
                    redsav = rednew
                }

                // Return if the reduction is zero. Otherwise, set the sine and cosine
                // of the angle of the alternative iteration, and calculate SDEC.
                if (isav < 0) {
                    state = 190
                    continue
                }
                if (isav < iu) {
                    temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext)
                    angt = angbd * (isav + HALF * temp) / iu
                }
                cth = (ONE - angt * angt) / (ONE + angt * angt)
                sth = (angt + angt) / (ONE + angt * angt)
                temp = shs + angt * (angt * dhd - dhs - dhs)
                sdec = sth * (angt * dredg - sredg - HALF * sth * temp)
                if (sdec <= ZERO) {
                    state = 190
                    continue
                }

                // Update GNEW, D and HRED. If the angle of the alternative iteration
                // is restricted by a bound on a free variable, that variable is fixed
                // at the bound.
                dredg = ZERO
                gredsq = ZERO
                for (i in 0 until n) {
                    gnew.setEntry(i, gnew.getEntry(i) + (cth - ONE) * hred.getEntry(i) + sth * hs.getEntry(i))
                    if (xbdi.getEntry(i) == ZERO) {
                        trialStepPoint!!.setEntry(i, cth * trialStepPoint!!.getEntry(i) + sth * s.getEntry(i))
                        dredg += trialStepPoint!!.getEntry(i) * gnew.getEntry(i)
                        // Computing 2nd power
                        val d1 = gnew.getEntry(i)
                        gredsq += d1 * d1
                    }
                    hred.setEntry(i, cth * hred.getEntry(i) + sth * hs.getEntry(i))
                }
                qred += sdec
                if (iact >= 0 && isav == iu) {
                    ++nact
                    xbdi.setEntry(iact, xsav)
                    state = 100
                    continue
                }

                // If SDEC is sufficiently small, then RETURN after setting XNEW to
                // XOPT+D, giving careful attention to the bounds.
                if (sdec > qred * .01) {
                    state = 120
                    continue
                }
                state = 190
            }
            if (state == 190) {
                printState(190)
                dsq = ZERO
                for (i in 0 until n) {
                    val min = FastMath.min(
                        trustRegionCenterOffset!!.getEntry(i) + trialStepPoint!!.getEntry(i),
                        upperDifference!!.getEntry(i)
                    )
                    newPoint!!.setEntry(i, FastMath.max(min, lowerDifference!!.getEntry(i)))
                    if (xbdi.getEntry(i) == MINUS_ONE) {
                        newPoint!!.setEntry(i, lowerDifference!!.getEntry(i))
                    }
                    if (xbdi.getEntry(i) == ONE) {
                        newPoint!!.setEntry(i, upperDifference!!.getEntry(i))
                    }
                    trialStepPoint!!.setEntry(i, newPoint!!.getEntry(i) - trustRegionCenterOffset!!.getEntry(i))
                    val d1 = trialStepPoint!!.getEntry(i)
                    dsq += d1 * d1
                }
                return doubleArrayOf(dsq, crvmin)
            }
            if(state == 210) {
                printState(210) // XXX
                var ih = 0
                for (j in 0 until n) {
                    hs.setEntry(j, ZERO)
                    for (i in 0..j) {
                        if (i < j) {
                            hs.setEntry(
                                j,
                                hs.getEntry(j) + modelSecondDerivativesValues!!.getEntry(ih) * s.getEntry(i)
                            )
                        }
                        hs.setEntry(i, hs.getEntry(i) + modelSecondDerivativesValues!!.getEntry(ih) * s.getEntry(j))
                        ih++
                    }
                }
                val tmp = interpolationPoints!!.operate(s).ebeMultiply(modelSecondDerivativesParameters)
                for (k in 0 until npt) {
                    if (modelSecondDerivativesParameters!!.getEntry(k) != ZERO) {
                        for (i in 0 until n) {
                            hs.setEntry(i, hs.getEntry(i) + tmp.getEntry(k) * interpolationPoints!!.getEntry(k, i))
                        }
                    }
                }
                if (crvmin != ZERO) {
                    state = 50
                    continue
                }
                if (iterc > itcsav) {
                    state = 150
                    continue
                }
                for (i in 0 until n) {
                    hred.setEntry(i, hs.getEntry(i))
                }
                state = 120
                continue
            } else {
                throw MathIllegalStateException(LocalizedFormats.SIMPLE_MESSAGE, "trsbox")
            }
        }
    } // trsbox
    // ----------------------------------------------------------------------------------------
    /**
     * The arrays BMAT and ZMAT are updated, as required by the new position
     * of the interpolation point that has the index KNEW. The vector VLAG has
     * N+NPT components, set on entry to the first NPT and last N components
     * of the product Hw in equation (4.11) of the Powell (2006) paper on
     * NEWUOA. Further, BETA is set on entry to the value of the parameter
     * with that name, and DENOM is set to the denominator of the updating
     * formula. Elements of ZMAT may be treated as zero if their moduli are
     * at most ZTEST. The first NDIM elements of W are used for working space.
     * @param beta
     * @param denom
     * @param knew
     */
    private fun update(
        beta: Double,
        denom: Double,
        knew: Int
    ) {
        printMethod() // XXX

        val n = currentBest!!.dimension
        val npt = numberOfInterpolationPoints
        val nptm = npt - n - 1

        // XXX Should probably be split into two arrays.
        val work = ArrayRealVector(npt + n)

        var ztest = ZERO
        for (k in 0 until npt) {
            for (j in 0 until nptm) {
                // Computing MAX
                ztest = FastMath.max(ztest, FastMath.abs(zMatrix!!.getEntry(k, j)))
            }
        }
        ztest *= 1e-20

        // Apply the rotations that put zeros in the KNEW-th row of ZMAT.
        for (j in 1 until nptm) {
            val d1 = zMatrix!!.getEntry(knew, j)
            if (FastMath.abs(d1) > ztest) {
                // Computing 2nd power
                val d2 = zMatrix!!.getEntry(knew, 0)
                // Computing 2nd power
                val d3 = zMatrix!!.getEntry(knew, j)
                val d4 = FastMath.sqrt(d2 * d2 + d3 * d3)
                val d5 = zMatrix!!.getEntry(knew, 0) / d4
                val d6 = zMatrix!!.getEntry(knew, j) / d4
                for (i in 0 until npt) {
                    val d7 = d5 * zMatrix!!.getEntry(i, 0) + d6 * zMatrix!!.getEntry(i, j)
                    zMatrix!!.setEntry(i, j, d5 * zMatrix!!.getEntry(i, j) - d6 * zMatrix!!.getEntry(i, 0))
                    zMatrix!!.setEntry(i, 0, d7)
                }
            }
            zMatrix!!.setEntry(knew, j, ZERO)
        }

        // Put the first NPT components of the KNEW-th column of HLAG into W,
        // and calculate the parameters of the updating formula.
        for (i in 0 until npt) {
            work.setEntry(i, zMatrix!!.getEntry(knew, 0) * zMatrix!!.getEntry(i, 0))
        }
        val alpha = work.getEntry(knew)
        val tau = lagrangeValuesAtNewPoint!!.getEntry(knew)
        lagrangeValuesAtNewPoint!!.setEntry(knew, lagrangeValuesAtNewPoint!!.getEntry(knew) - ONE)

        // Complete the updating of ZMAT.
        val sqrtDenom = FastMath.sqrt(denom)
        val d1 = tau / sqrtDenom
        val d2 = zMatrix!!.getEntry(knew, 0) / sqrtDenom
        for (i in 0 until npt) {
            zMatrix!!.setEntry(
                i, 0,
                d1 * zMatrix!!.getEntry(i, 0) - d2 * lagrangeValuesAtNewPoint!!.getEntry(i)
            )
        }

        // Finally, update the matrix BMAT.
        for (j in 0 until n) {
            val jp = npt + j
            work.setEntry(jp, bMatrix!!.getEntry(knew, j))
            val d3 = (alpha * lagrangeValuesAtNewPoint!!.getEntry(jp) - tau * work.getEntry(jp)) / denom
            val d4 = (-beta * work.getEntry(jp) - tau * lagrangeValuesAtNewPoint!!.getEntry(jp)) / denom
            for (i in 0..jp) {
                bMatrix!!.setEntry(
                    i, j,
                    bMatrix!!.getEntry(i, j) + d3 * lagrangeValuesAtNewPoint!!.getEntry(i) + d4 * work.getEntry(i)
                )
                if (i >= npt) {
                    bMatrix!!.setEntry(jp, i - npt, bMatrix!!.getEntry(i, j))
                }
            }
        }
    } // update

    /**
     * Performs validity checks.
     *
     * @param lowerBound Lower bounds (constraints) of the objective variables.
     * @param upperBound Upperer bounds (constraints) of the objective variables.
     */
    private fun setup(
        lowerBound: DoubleArray,
        upperBound: DoubleArray
    ) {
        printMethod() // XXX

        val init = startPoint
        val dimension = init.size

        // Check problem dimension.
        if (dimension < MINIMUM_PROBLEM_DIMENSION) {
            throw NumberIsTooSmallException(dimension, MINIMUM_PROBLEM_DIMENSION, true)
        }
        // Check number of interpolation points.
        val nPointsInterval = intArrayOf(dimension + 2, (dimension + 2) * (dimension + 1) / 2)
        if (numberOfInterpolationPoints < nPointsInterval[0] ||
            numberOfInterpolationPoints > nPointsInterval[1]
        ) {
            throw OutOfRangeException(
                LocalizedFormats.NUMBER_OF_INTERPOLATION_POINTS,
                numberOfInterpolationPoints,
                nPointsInterval[0],
                nPointsInterval[1]
            )
        }

        // Initialize bound differences.
        boundDifference = DoubleArray(dimension)

        val requiredMinDiff = 2 * initialTrustRegionRadius
        var minDiff = Double.POSITIVE_INFINITY
        for (i in 0 until dimension) {
            boundDifference[i] = upperBound[i] - lowerBound[i]
            minDiff = FastMath.min(minDiff, boundDifference[i])
        }
        if (minDiff < requiredMinDiff) {
            initialTrustRegionRadius = minDiff / 3.0
        }

        // Initialize the data structures used by the "bobyqa" method.
        bMatrix = Array2DRowRealMatrix(
            dimension + numberOfInterpolationPoints,
            dimension
        )
        zMatrix = Array2DRowRealMatrix(
            numberOfInterpolationPoints,
            numberOfInterpolationPoints - dimension - 1
        )
        interpolationPoints = Array2DRowRealMatrix(
            numberOfInterpolationPoints,
            dimension
        )
        originShift = ArrayRealVector(dimension)
        fAtInterpolationPoints = ArrayRealVector(numberOfInterpolationPoints)
        trustRegionCenterOffset = ArrayRealVector(dimension)
        gradientAtTrustRegionCenter = ArrayRealVector(dimension)
        lowerDifference = ArrayRealVector(dimension)
        upperDifference = ArrayRealVector(dimension)
        modelSecondDerivativesParameters = ArrayRealVector(numberOfInterpolationPoints)
        newPoint = ArrayRealVector(dimension)
        alternativeNewPoint = ArrayRealVector(dimension)
        trialStepPoint = ArrayRealVector(dimension)
        lagrangeValuesAtNewPoint = ArrayRealVector(dimension + numberOfInterpolationPoints)
        modelSecondDerivativesValues = ArrayRealVector(dimension * (dimension + 1) / 2)
    }

    /**
     * Marker for code paths that are not explored with the current unit tests.
     * If the path becomes explored, it should just be removed from the code.
     */
    private object PathIsExploredException : RuntimeException() {
        /** Serializable UID.  */
        private const val serialVersionUID = 745350979634801853L

        /** Message string.  */
        private const val PATH_IS_EXPLORED = "If this exception is thrown, just remove it from the code"
    }

    companion object {
        /** Minimum dimension of the problem: {@value}  */
        const val MINIMUM_PROBLEM_DIMENSION = 2

        /** Default value for [.initialTrustRegionRadius]: {@value} .  */
        const val DEFAULT_INITIAL_RADIUS = 10.0

        /** Default value for [.stoppingTrustRegionRadius]: {@value} .  */
        const val DEFAULT_STOPPING_RADIUS = 1E-8

        /** Constant 0.  */
        private const val ZERO = 0.0

        /** Constant 1.  */
        private const val ONE = 1.0

        /** Constant 2.  */
        private const val TWO = 2.0

        /** Constant 10.  */
        private const val TEN = 10.0

        /** Constant 16.  */
        private const val SIXTEEN = 16.0

        /** Constant 250.  */
        private const val TWO_HUNDRED_FIFTY = 250.0

        /** Constant -1.  */
        private const val MINUS_ONE = -ONE

        /** Constant 1/2.  */
        private const val HALF = ONE / 2

        /** Constant 1/4.  */
        private const val ONE_OVER_FOUR = ONE / 4

        /** Constant 1/8.  */
        private const val ONE_OVER_EIGHT = ONE / 8

        /** Constant 1/10.  */
        private const val ONE_OVER_TEN = ONE / 10

        /** Constant 1/1000.  */
        private const val ONE_OVER_A_THOUSAND = ONE / 1000

        // XXX utility for figuring out call sequence.
        private fun caller(n: Int): String {
            val t = Throwable()
            val elements = t.stackTrace
            val e = elements[n]
            return e.methodName + " (at line " + e.lineNumber + ")"
        }

        // XXX utility for figuring out call sequence.
        private fun printState(s: Int) {
            //        System.out.println(caller(2) + ": state " + s);
        }

        // XXX utility for figuring out call sequence.
        private fun printMethod() {
            //        System.out.println(caller(2));
        }
    }
}