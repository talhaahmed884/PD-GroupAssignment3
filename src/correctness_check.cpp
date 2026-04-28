/*
 * correctness_check.cpp
 *
 * Standalone correctness validator for all four MM implementations.
 * Compile: g++ -O2 -fopenmp -std=c++17 src/correctness_check.cpp -o correctness_check
 *
 * Usage (called by run_correctness.sh — passes stdout from ./mm via pipe):
 *   ./mm <algo> <m> <n> <q> <P> | ./correctness_check <m> <n> <q>
 *
 * What it does:
 *   1. Seeds RNG with 42 and generates the SAME A(m×n), B(n×q) the main binary uses
 *      (teammates must also seed with 42 — document this in interface_contract.md)
 *   2. Computes reference C_ref using a serial triple-loop
 *   3. Reads the test output from stdin, parsing lines like: C[i][j]=<value>
 *   4. Compares element-wise with tolerance 1e-9
 *   5. Prints PASS or FAIL (with first 5 mismatches); exits 0=pass, 1=fail
 *
 * NOTE: If teammates do not print the C matrix, this checker cannot validate.
 * Ask them to add a "--print-matrix" flag or a compile-time #define.
 * The run_correctness.sh script already warns when C[i][j] lines are absent.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <map>
#include <vector>

// ============================================================
// CONFIGURATION
// ============================================================
static const double TOL = 1e-9;
static const int RNG_SEED = 42;
static const int MAX_MISMATCHES_SHOWN = 5;

// ============================================================
// MATRIX HELPERS
// ============================================================
static double *alloc_matrix(int rows, int cols)
{
    double *m = new double[rows * cols]();
    return m;
}

static void free_matrix(double *m)
{
    delete[] m;
}

static double &mat(double *m, int cols, int i, int j)
{
    return m[i * cols + j];
}

// ============================================================
// REFERENCE: fill matrices with the same RNG sequence teammates use
// ============================================================
static void fill_matrix(double *m, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        m[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

// ============================================================
// REFERENCE: serial triple-loop MM
// ============================================================
static void serial_mm(double *A, double *B, double *C, int m, int n, int q)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < q; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += mat(A, n, i, k) * mat(B, q, k, j);
            mat(C, q, i, j) = sum;
        }
}

// ============================================================
// PARSE: read "C[i][j]=value" lines from stdin
// Returns false if no C[i][j] lines were found at all
// ============================================================
static bool parse_matrix_output(std::map<std::pair<int, int>, double> &out)
{
    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), stdin))
    {
        int ri, ci;
        double val;
        if (sscanf(line, "C[%d][%d]=%lf", &ri, &ci, &val) == 3)
        {
            out[{ri, ci}] = val;
            count++;
        }
        // Lines that don't match (like "TIME: ...") are silently ignored
    }
    return count > 0;
}

// ============================================================
// COMPARE
// ============================================================
static bool compare(double *C_ref,
                    const std::map<std::pair<int, int>, double> &C_test,
                    int m, int q)
{
    int mismatch_count = 0;
    bool passed = true;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < q; j++)
        {
            double ref = mat(C_ref, q, i, j);

            auto it = C_test.find({i, j});
            if (it == C_test.end())
            {
                if (mismatch_count < MAX_MISMATCHES_SHOWN)
                {
                    fprintf(stderr, "  MISSING C[%d][%d]  (expected %.9f)\n", i, j, ref);
                }
                mismatch_count++;
                passed = false;
                continue;
            }

            double test = it->second;
            double diff = fabs(ref - test);
            if (diff > TOL)
            {
                if (mismatch_count < MAX_MISMATCHES_SHOWN)
                {
                    fprintf(stderr, "  MISMATCH C[%d][%d]: got %.9f  expected %.9f  diff=%.2e\n",
                            i, j, test, ref, diff);
                }
                mismatch_count++;
                passed = false;
            }
        }
    }

    if (mismatch_count > MAX_MISMATCHES_SHOWN)
    {
        fprintf(stderr, "  ... and %d more mismatches (not shown)\n",
                mismatch_count - MAX_MISMATCHES_SHOWN);
    }
    return passed;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: ./mm <algo> <m> <n> <q> <P> | ./correctness_check <m> <n> <q>\n");
        return 2;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int q = atoi(argv[3]);

    if (m <= 0 || n <= 0 || q <= 0)
    {
        fprintf(stderr, "correctness_check: invalid dimensions %d %d %d\n", m, n, q);
        return 2;
    }

    // Generate reference matrices with the same seed teammates use
    srand(RNG_SEED);
    double *A = alloc_matrix(m, n);
    double *B = alloc_matrix(n, q);
    double *C_ref = alloc_matrix(m, q);

    fill_matrix(A, m, n);
    fill_matrix(B, n, q);
    serial_mm(A, B, C_ref, m, n, q);

    // Parse test output from stdin
    std::map<std::pair<int, int>, double> C_test;
    bool has_output = parse_matrix_output(C_test);

    if (!has_output)
    {
        fprintf(stderr, "correctness_check: no C[i][j]=val lines found in stdin.\n");
        fprintf(stderr, "  The implementation must print matrix C in this format:\n");
        fprintf(stderr, "    for each i,j: printf(\"C[%%d][%%d]=%%f\\n\", i, j, C[i][j]);\n");
        fprintf(stderr, "  Ask teammate to add a --print-matrix flag or conditional output.\n");
        free_matrix(A);
        free_matrix(B);
        free_matrix(C_ref);
        return 1;
    }

    bool passed = compare(C_ref, C_test, m, q);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C_ref);

    if (passed)
    {
        // run_correctness.sh prints the PASS label; keep stdout clean for piping
        return 0;
    }
    else
    {
        return 1;
    }
}
