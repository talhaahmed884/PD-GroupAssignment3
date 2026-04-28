/*
 * mm.cpp — Matrix Multiplication: main entry point
 *
 * Usage (serial):  mpirun -np 1  ./mm ser <m> <n> <q> <P>
 * Usage (MPI 1-D): mpirun -np <P> ./mm 1d  <m> <n> <q> <P>
 * Usage (MPI 2-D): mpirun -np <P> ./mm 2d  <m> <n> <q> <P>
 *   algo : ser | 1d | 2d
 *   m n q : dimensions  A(m×n) × B(n×q) = C(m×q)
 *   P    : number of MPI processes (perfect square for 2d)
 *
 * Output:
 *   C[i][j]=<val>   only when MM_PRINT_MATRIX=1  (used by run_correctness.sh)
 *   TIME: <seconds>  always                       (used by run_experiments.sh)
 *
 * Compile:  mpicxx -O2 -std=c++17 src/mm.cpp -o mm
 *       or: make
 *
 * ── Interface contract ────────────────────────────────────────────────────────
 * MM-1D initial layout:
 *   A: row strip i → process i  (m/P rows × n cols each)
 *   B: row strip i → process i  (n/P rows × q cols each)
 *   Ring-shift passes each B strip through all ranks so every rank
 *   accumulates its full partial product.
 *
 * MM-2D initial layout (image §1.3):
 *   A(m×n): process P_{i,j} owns block rows [i·mb..(i+1)·mb-1],
 *                                  block cols [j·nb..(j+1)·nb-1]
 *   B(n×q): process P_{i,j} owns block rows [j·nb..(j+1)·nb-1],
 *                                  block cols [i·qb..(i+1)·qb-1]
 *   where mb=m/√P, nb=n/√P, qb=q/√P.
 *   B is redistributed (transpose swap) to standard B_{i,j}→P_{i,j} layout,
 *   then SUMMA broadcasts √P panels to accumulate C.
 * ──────────────────────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mpi.h>

// ─── Matrix helpers (row-major, C++ new/delete) ──────────────────────────────

static double *alloc_mat(int rows, int cols)
{
    return new double[rows * cols]();
}

static void free_mat(double *m)
{
    delete[] m;
}

// Access element [i][j] of a matrix stored with 'stride' columns
static inline double &E(double *mat, int stride, int i, int j)
{
    return mat[i * stride + j];
}

static void fill_mat(double *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
        mat[i] = static_cast<double>(rand()) / RAND_MAX;
}

// ─── 1.2  MM-1D helpers (partner contribution — do not modify) ───────────────
// Note: alloc_matrix uses calloc; release with free(), not free_mat/delete[].

static void print_matrix(const char *name, double *M, int rows, int cols)
{
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            printf("%6.1f ", M[i * cols + j]);
        printf("\n");
    }
    printf("\n");
}

static double *alloc_matrix(int rows, int cols)
{
    double *mat = (double *)calloc(rows * cols, sizeof(double));
    if (!mat)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return mat;
}

// compute_partial_c: accumulate C_local += A_local[:,k_start..k_start+n_local-1] * B_current
// A_local is stored with row stride n (its full allocated width, not just n_local).
// Also used by mm_2d: call with n=nb and k_start=0 when A_local is a pre-extracted panel.
static void compute_partial_c(double *A_local, double *B_current, double *C_local,
                              int m_local, int n_local, int n, int q, int k_start)
{
    for (int i = 0; i < m_local; i++)
        for (int k = 0; k < n_local; k++)
        {
            double a = A_local[i * n + (k_start + k)];
            for (int j = 0; j < q; j++)
                C_local[i * q + j] += a * B_current[k * q + j];
        }
}

static void ring_shift_B(double *B_current, double *B_recv, int count, int world_rank, int world_size, MPI_Comm comm)
{
    int right = (world_rank + 1) % world_size;
    int left = (world_rank - 1 + world_size) % world_size;

    // even ranks send first, odd ranks receive first to avoid deadlock
    if (world_rank % 2 == 0)
    {
        MPI_Send(B_current, count, MPI_DOUBLE, right, 0, comm);
        MPI_Recv(B_recv, count, MPI_DOUBLE, left, 0, comm, MPI_STATUS_IGNORE);
    }
    else
    {
        MPI_Recv(B_recv, count, MPI_DOUBLE, left, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(B_current, count, MPI_DOUBLE, right, 0, comm);
    }
}

// ─── 1.1  Serial MM (MM-ser) ─────────────────────────────────────────────────

static void mm_ser(double *A, double *B, double *C, int m, int n, int q)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < q; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += E(A, n, i, k) * E(B, q, k, j);
            E(C, q, i, j) = sum;
        }
}

// ─── 1.2  MM-1D (MPI, 1-D row strip + ring-shift B) ─────────────────────────
//
// Rank 0 holds the full A (m×n) and B (n×q); all others receive nullptr.
// Each rank gets m/P rows of A (full n cols) and n/P rows of B (full q cols).
// P ring-shift steps rotate B strips so every rank accumulates its full C strip.

static void mm_1d(double *A, double *B, double *C, int m, int n, int q, int P)
{
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (m % world_size != 0 || n % world_size != 0)
    {
        if (world_rank == 0)
        {
            fprintf(stderr, "Matrix dimensions must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    int m_local = m / world_size;
    int n_local = n / world_size;
    int q_local = q / world_size;
    int strip_count = n_local * q;

    (void)q_local; // computed for symmetry; not used directly in the ring loop
    (void)P;

    // ── Scatter A: each rank receives m_local rows × full n cols ─────────────
    double *A_local = alloc_matrix(m_local, n);

    if (world_rank == 0)
    {
        memcpy(A_local, A, m_local * n * sizeof(double));
        for (int i = 1; i < world_size; i++)
        {
            MPI_Send(A + i * m_local * n, m_local * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(A_local, m_local * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // ── Scatter B: each rank receives n_local rows × full q cols ─────────────
    double *B_current = alloc_matrix(n_local, q);

    if (world_rank == 0)
    {
        memcpy(B_current, B, strip_count * sizeof(double));
        for (int i = 1; i < world_size; i++)
        {
            MPI_Send(B + i * strip_count, strip_count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(B_current, strip_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double *C_local = alloc_matrix(m_local, q);
    double *B_recv = alloc_matrix(n_local, q);

    int b_owner = world_rank;

    // ── Ring algorithm: P steps, each using A columns matching the B strip ────
    for (int step = 0; step < world_size; step++)
    {
        int k_start = b_owner * n_local;
        compute_partial_c(A_local, B_current, C_local, m_local, n_local, n, q, k_start);

        if (step < world_size - 1)
        { // no need to shift after the last step
            ring_shift_B(B_current, B_recv, strip_count, world_rank, world_size, MPI_COMM_WORLD);
            double *temp = B_current;
            B_current = B_recv;
            B_recv = temp;
            b_owner = (b_owner - 1 + world_size) % world_size; // update owner for next step
        }
    }

    // ── Gather C strips into rank 0's output matrix ───────────────────────────
    if (world_rank == 0)
    {
        memcpy(C, C_local, m_local * q * sizeof(double));

        for (int i = 1; i < world_size; i++)
        {
            MPI_Recv(C + i * m_local * q, m_local * q,
                     MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        MPI_Send(C_local, m_local * q, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    free(A_local);
    free(B_current);
    free(B_recv);
    free(C_local);
}

// ─── 1.3  MM-2D (MPI, √P × √P process grid, SUMMA) ──────────────────────────
//
// Initial layout per §1.3 image:
//   A: block (i,j) → P_{i,j}          (row-block i, col-block j of A)
//   B: block (r,c) in B's grid → P_{c,r}
//      i.e. P_{i,j} initially holds B_{j,i} (row-block j, col-block i of B)
//
// We first redistribute B so that P_{i,j} holds B_{i,j} (standard SUMMA
// layout), then run √P rounds of panel broadcasts.
// The inner local multiply reuses compute_partial_c (call with n=nb, k_start=0
// so the panel is treated as a self-contained block with stride nb).

static void mm_2d(double *A, double *B, double *C, int m, int n, int q, int P)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int sp = (int)round(sqrt((double)P)); // √P

    // ── 2-D Cartesian communicator (reorder=0 keeps ranks stable) ────────────
    int dims[2] = {sp, sp};
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int pi = coords[0]; // this process's row in the grid
    int pj = coords[1]; // this process's col in the grid

    // Block dimensions (P divides m, n, q evenly for the supported cases)
    int mb = m / sp; // A/C row block height
    int nb = n / sp; // A/B inner block width/height
    int qb = q / sp; // B/C col block width

    // ── Allocate local storage ────────────────────────────────────────────────
    double *locA = alloc_mat(mb, nb); // will hold A_{pi,pj}
    double *locB = alloc_mat(nb, qb); // will hold B_{pi,pj} after redistribution
    double *locC = alloc_mat(mb, qb); // accumulates C_{pi,pj}

    // ── Scatter A from rank 0 ─────────────────────────────────────────────────
    // P_{r,c} receives A rows [r·mb..(r+1)·mb-1], A cols [c·nb..(c+1)·nb-1]
    if (rank == 0)
    {
        for (int r = 0; r < sp; r++)
        {
            for (int c = 0; c < sp; c++)
            {
                double *buf = alloc_mat(mb, nb);
                for (int i = 0; i < mb; i++)
                    for (int j = 0; j < nb; j++)
                        buf[i * nb + j] = E(A, n, r * mb + i, c * nb + j);

                int dst_coords[2] = {r, c};
                int dst;
                MPI_Cart_rank(cart, dst_coords, &dst);

                if (dst == 0)
                    memcpy(locA, buf, mb * nb * sizeof(double));
                else
                    MPI_Send(buf, mb * nb, MPI_DOUBLE, dst, 10, cart);

                free_mat(buf);
            }
        }
    }
    else
    {
        MPI_Recv(locA, mb * nb, MPI_DOUBLE, 0, 10, cart, MPI_STATUS_IGNORE);
    }

    // ── Scatter B from rank 0 ─────────────────────────────────────────────────
    // Per the §1.3 image, P_{r,c} initially holds B_{c,r}:
    //   B rows [c·nb..(c+1)·nb-1], B cols [r·qb..(r+1)·qb-1]
    if (rank == 0)
    {
        for (int r = 0; r < sp; r++)
        {
            for (int c = 0; c < sp; c++)
            {
                double *buf = alloc_mat(nb, qb);
                for (int i = 0; i < nb; i++)
                    for (int j = 0; j < qb; j++)
                        buf[i * qb + j] = E(B, q, c * nb + i, r * qb + j);

                int dst_coords[2] = {r, c};
                int dst;
                MPI_Cart_rank(cart, dst_coords, &dst);

                if (dst == 0)
                    memcpy(locB, buf, nb * qb * sizeof(double));
                else
                    MPI_Send(buf, nb * qb, MPI_DOUBLE, dst, 11, cart);

                free_mat(buf);
            }
        }
    }
    else
    {
        MPI_Recv(locB, nb * qb, MPI_DOUBLE, 0, 11, cart, MPI_STATUS_IGNORE);
    }

    // ── Redistribute B: P_{i,j} holds B_{j,i}; swap with P_{j,i} → B_{i,j} ─
    // Both blocks are the same shape (nb×qb), so Sendrecv works directly.
    // Diagonal processes (pi==pj) send to themselves — MPI_Sendrecv handles it.
    {
        int swap_coords[2] = {pj, pi};
        int swap_rank;
        MPI_Cart_rank(cart, swap_coords, &swap_rank);

        double *tmp = alloc_mat(nb, qb);
        MPI_Sendrecv(locB, nb * qb, MPI_DOUBLE, swap_rank, 20,
                     tmp, nb * qb, MPI_DOUBLE, swap_rank, 20,
                     cart, MPI_STATUS_IGNORE);
        memcpy(locB, tmp, nb * qb * sizeof(double));
        free_mat(tmp);
    }
    // P_{i,j} now holds B_{i,j}: B rows [pi·nb..(pi+1)·nb-1], cols [pj·qb..(pj+1)·qb-1]

    // ── Row and column sub-communicators ──────────────────────────────────────
    // row_comm: all processes with the same pi, ordered by pj (rank == pj)
    // col_comm: all processes with the same pj, ordered by pi (rank == pi)
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart, pi, pj, &row_comm);
    MPI_Comm_split(cart, pj, pi, &col_comm);

    // ── SUMMA ─────────────────────────────────────────────────────────────────
    double *panA = alloc_mat(mb, nb);
    double *panB = alloc_mat(nb, qb);

    for (int k = 0; k < sp; k++)
    {
        // Broadcast A_{pi,k} from the process with pj==k (rank k in row_comm)
        if (pj == k)
            memcpy(panA, locA, mb * nb * sizeof(double));
        MPI_Bcast(panA, mb * nb, MPI_DOUBLE, k, row_comm);

        // Broadcast B_{k,pj} from the process with pi==k (rank k in col_comm)
        if (pi == k)
            memcpy(panB, locB, nb * qb * sizeof(double));
        MPI_Bcast(panB, nb * qb, MPI_DOUBLE, k, col_comm);

        // Accumulate locC += panA * panB  (n=nb since panA's row stride is nb)
        compute_partial_c(panA, panB, locC, mb, nb, nb, qb, 0);
    }

    // ── Gather C to rank 0 ────────────────────────────────────────────────────
    // P_{r,c} owns C_{r,c}: rows [r·mb..(r+1)·mb-1], cols [c·qb..(c+1)·qb-1]
    if (rank == 0)
    {
        for (int i = 0; i < mb; i++)
            for (int j = 0; j < qb; j++)
                E(C, q, i, j) = locC[i * qb + j];

        for (int r = 0; r < sp; r++)
        {
            for (int c = 0; c < sp; c++)
            {
                int src_coords[2] = {r, c};
                int src;
                MPI_Cart_rank(cart, src_coords, &src);
                if (src == 0)
                    continue;

                double *buf = alloc_mat(mb, qb);
                MPI_Recv(buf, mb * qb, MPI_DOUBLE, src, 30, cart, MPI_STATUS_IGNORE);
                for (int i = 0; i < mb; i++)
                    for (int j = 0; j < qb; j++)
                        E(C, q, r * mb + i, c * qb + j) = buf[i * qb + j];
                free_mat(buf);
            }
        }
    }
    else
    {
        MPI_Send(locC, mb * qb, MPI_DOUBLE, 0, 30, cart);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    free_mat(locA);
    free_mat(locB);
    free_mat(locC);
    free_mat(panA);
    free_mat(panB);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart);
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 6)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Usage: mpirun -np <P> ./mm <algo> <m> <n> <q> <P>\n");
            fprintf(stderr, "  algo: ser | 1d | 2d\n");
        }
        MPI_Finalize();
        return 2;
    }

    const char *algo = argv[1];
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int q = atoi(argv[4]);
    int P = atoi(argv[5]);

    if (m <= 0 || n <= 0 || q <= 0 || P <= 0)
    {
        if (rank == 0)
            fprintf(stderr, "mm: invalid dimensions or process count\n");
        MPI_Finalize();
        return 2;
    }

    // Rank 0 owns the full matrices; others receive their blocks inside each algo
    double *A = nullptr;
    double *B = nullptr;
    double *C = nullptr;

    if (rank == 0)
    {
        srand(42);
        A = alloc_mat(m, n);
        B = alloc_mat(n, q);
        C = alloc_mat(m, q);
        fill_mat(A, m, n);
        fill_mat(B, n, q);
    }

    // ── Time only the multiplication ─────────────────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    if (strcmp(algo, "ser") == 0)
    {
        if (rank == 0)
            mm_ser(A, B, C, m, n, q);
    }
    else if (strcmp(algo, "1d") == 0)
    {
        mm_1d(A, B, C, m, n, q, P);
    }
    else if (strcmp(algo, "2d") == 0)
    {
        mm_2d(A, B, C, m, n, q, P);
    }
    else
    {
        if (rank == 0)
            fprintf(stderr, "mm: unknown algorithm '%s'\n", algo);
        if (rank == 0)
        {
            free_mat(A);
            free_mat(B);
            free_mat(C);
        }
        MPI_Finalize();
        return 2;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    // ─────────────────────────────────────────────────────────────────────────

    if (rank == 0)
    {
        // Print result matrix when MM_PRINT_MATRIX=1 (set by run_correctness.sh)
        const char *print_env = getenv("MM_PRINT_MATRIX");
        if (print_env != nullptr && print_env[0] == '1')
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < q; j++)
                    printf("C[%d][%d]=%.15f\n", i, j, E(C, q, i, j));
        }

        // Always print timing — parsed by run_experiments.sh
        printf("TIME: %.9f\n", t_end - t_start);

        free_mat(A);
        free_mat(B);
        free_mat(C);
    }

    MPI_Finalize();
    return 0;
}
