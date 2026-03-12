#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include "locality_aware.h"

// Serial SpMV b = alpha*A*x + beta*b
void spmv(double alpha, Mat& A, std::vector<double>& x,
        double beta, std::vector<double>& b)
{
    double sum;
    int start, end;

    for (int i = 0; i < A.n_rows; i++)
    {
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        sum = 0;
        for (int j = start; j < end; j++)
        {
            sum += A.data[j] * x[A.col_idx[j]];
        }
        b[i] = alpha * sum + beta * b[i];
    }
}

// Parallel SpMV b = alpha*A*x + beta*b 
void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b)
{
    int proc, start, end;
    int tag = 0;
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc  = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i];
        end   = A.recv_comm.ptr[i + 1];
        MPI_Irecv(&(recvbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.recv_comm.req[i]));
    }

    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc  = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end   = A.send_comm.ptr[i + 1];
        for (int j = start; j < end; j++)
            sendbuf[j] = x[A.send_comm.idx[j]];
        MPI_Isend(&(sendbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.send_comm.req[i]));
    }

    spmv(alpha, A.on_proc, x, beta, b);

    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    if (A.send_comm.n_msgs)
    {
        MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    spmv(alpha, A.off_proc, recvbuf, 1.0, b);

}

void axpy(double alpha, std::vector<double>& x, std::vector<double>& y)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = x[i] + alpha*y[i];
}

void scale(double alpha, std::vector<double>& x)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = alpha*x[i];
}

void CG_persistent(ParMat& A, std::vector<double>& x, std::vector<double>& b)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);
    std::vector<double> res;

    // Setup persistent allreduces
    double local_sum, global_sum;
    MPIL_Request* mpil_req;
    MPIL_Comm* mpil_comm;
    MPIL_Comm_init(&mpil_comm, MPI_COMM_WORLD);
    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);
    MPIL_Allreduce_init(&local_sum, &global_sum, 1, MPI_DOUBLE,
           MPI_SUM, mpil_comm, mpil_info, &mpil_req); 

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r, tol = 1e-6;
    int max_iter = ((int)(1.3*b.size())) + 2;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r);

    // p0 = r0
    p = r;

    // Find initial (r, r) and residual
    local_sum = 0;
    for (int i = 0; i < r.size(); i++)
        local_sum += r[i] * r[i];
    MPIL_Start(mpil_req);
    MPIL_Wait(mpil_req, MPI_STATUS_IGNORE);
    rr_inner = global_sum;

    norm_r = sqrt(rr_inner);
    res.push_back(norm_r);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, 0.0, Ap);
        local_sum = 0;
        for (int i = 0; i < Ap.size(); i++)
            local_sum += Ap[i] * p[i];
        MPIL_Start(mpil_req);
        MPIL_Wait(mpil_req, MPI_STATUS_IGNORE);
        App_inner = global_sum;
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rr_inner / App_inner;

        axpy(alpha, x, p);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            axpy(-1.0*alpha, r, Ap);
        }
        else
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r);
        }

        local_sum = 0;
        for (int i = 0; i < r.size(); i++)
            local_sum += r[i] * r[i];
        MPIL_Start(mpil_req);
        MPIL_Wait(mpil_req, MPI_STATUS_IGNORE);
        next_inner = global_sum;
        beta = next_inner / rr_inner;

        scale(beta, p);
        axpy(1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);

        res.push_back(norm_r);

        iter++;
    }

    if (rank == 0) 
    {
        if (iter == max_iter)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iteration required to converge\n", iter);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
    }


    MPIL_Request_free(&mpil_req);
    MPIL_Info_free(&mpil_info);
    MPIL_Comm_free(&mpil_comm);
}

void CG(ParMat& A, std::vector<double>& x, std::vector<double>& b)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);
    std::vector<double> res;

    // Setup persistent allreduces
    double local_sum, global_sum;
    MPIL_Request* mpil_req;
    MPIL_Comm* mpil_comm;
    MPIL_Comm_init(&mpil_comm, MPI_COMM_WORLD);

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r, tol = 1e-6;
    int max_iter = ((int)(1.3*b.size())) + 2;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r);

    // p0 = r0
    p = r;

    // Find initial (r, r) and residual
    local_sum = 0;
    for (int i = 0; i < r.size(); i++)
        local_sum += r[i] * r[i];
    MPIL_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
            mpil_comm);
    rr_inner = global_sum;

    norm_r = sqrt(rr_inner);
    res.push_back(norm_r);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, 0.0, Ap);
        local_sum = 0;
        for (int i = 0; i < Ap.size(); i++)
            local_sum += Ap[i] * p[i];
        MPIL_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                mpil_comm);
        App_inner = global_sum;

        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rr_inner / App_inner;

        axpy(alpha, x, p);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            axpy(-1.0*alpha, r, Ap);
        }
        else
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r);
        }

        local_sum = 0;
        for (int i = 0; i < r.size(); i++)
            local_sum += r[i] * r[i];
        MPIL_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                mpil_comm);
        next_inner = global_sum;
        beta = next_inner / rr_inner;

        scale(beta, p);
        axpy(1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);

        res.push_back(norm_r);

        iter++;
    }

    if (rank == 0) 
    {
        if (iter == max_iter)
            printf("Max Iterations Reached.\n");
        else
            printf("%d Iteration required to converge\n", iter);
        printf("2 Norm of Residual: %lg\n\n", norm_r);
    }


    MPIL_Comm_free(&mpil_comm);

}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double t0, tfinal;
    int n_iters = 1;

    const char* filename = "Dubcova2.pm";
    if (argc > 1)
    {
        filename = argv[1];
    }

    ParMat A;
    readParMatrix(filename, A);
    form_comm(A);
    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);

    // Set b to random values, x to 0
    srand(time(NULL) + rank);
    std::generate(x.begin(), x.end(), 
            [&](){ return (double)(rand()) / RAND_MAX; });
    spmv(1.0, A, x, 0.0, b);

    // PMPI
    MPIL_Set_allreduce_algorithm(ALLREDUCE_PMPI);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with PMPI Allreduce: %e\n", t0);

    // Recursive Doubling
    MPIL_Set_allreduce_algorithm(ALLREDUCE_RECURSIVE_DOUBLING);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with MPIL Recursive Doubling Allreduce: %e\n", t0);

    // Dissemination Node-Aware
    MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_LOC);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with MPIL Node-Aware Dissemination Allreduce: %e\n", t0);

    // Dissemination Locality-Aware
    MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_ML);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with MPIL Locality-Aware Dissemination Allreduce: %e\n", t0);

    // Dissemination High-Radix
    MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_RADIX);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with MPIL High-Radix Dissemination Allreduce: %e\n", t0);

    // Dissemination RMA
    MPIL_Set_allreduce_algorithm(ALLREDUCE_RMA);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with MPIL RMA Allreduce: %e\n", t0);

    MPI_Finalize();
}
