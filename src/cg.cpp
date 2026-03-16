#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include "locality_aware.h"
#include <math.h>

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

void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b, MPIL_Comm* mpil_comm)
{
    int proc, start, end;
    int tag = 0;
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);

    MPIL_Topo* mpil_topo;
    MPIL_Topo_init(A.recv_comm.n_msgs,
            A.recv_comm.procs.data(),
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs,
            A.send_comm.procs.data(),
            MPI_UNWEIGHTED,
            mpil_info,
            &mpil_topo);

    // Pack Send Buffer
    for (int i = 0; i < A.send_comm.size_msgs; i++)
        sendbuf[i] = x[A.send_comm.idx[i]];

    MPIL_Neighbor_alltoallv_topo(sendbuf.data(), 
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(),
            MPI_DOUBLE,
            recvbuf.data(),
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(),
            MPI_DOUBLE,
            mpil_topo,
            mpil_comm);

    spmv(alpha, A.on_proc, x, beta, b);

    spmv(alpha, A.off_proc, recvbuf, 1.0, b);

    MPIL_Info_free(&mpil_info);
    MPIL_Topo_free(&mpil_topo);
}


// Parallel SpMV b = alpha*A*x + beta*b 
void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b, MPIL_Comm* mpil_comm,
        MPIL_Request** req_ptr)
{
    int proc, start, end;
    int tag = 0;
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    if (*req_ptr == NULL)
    {
        MPIL_Info* mpil_info;
        MPIL_Info_init(&mpil_info);

        MPIL_Topo* mpil_topo;
        MPIL_Topo_init(A.recv_comm.n_msgs,
                A.recv_comm.procs.data(),
                MPI_UNWEIGHTED,
                A.send_comm.n_msgs,
                A.send_comm.procs.data(),
                MPI_UNWEIGHTED,
                mpil_info,
                &mpil_topo);

        std::vector<long> global_send_idx(A.send_comm.size_msgs);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
            global_send_idx[i] = A.send_comm.idx[i] + A.first_row;
        MPIL_Neighbor_alltoallv_init_ext_topo(sendbuf.data(), 
                A.send_comm.counts.data(),
                A.send_comm.ptr.data(),
                global_send_idx.data(),
                MPI_DOUBLE,
                recvbuf.data(),
                A.recv_comm.counts.data(),
                A.recv_comm.ptr.data(),
                A.off_proc_columns.data(),
                MPI_DOUBLE,
                mpil_topo,
                mpil_comm,
                mpil_info,
                req_ptr);

        MPIL_Info_free(&mpil_info);
        MPIL_Topo_free(&mpil_topo);
    }

    for (int i = 0; i < A.send_comm.size_msgs; i++)
        sendbuf[i] = x[A.send_comm.idx[i]];
    MPIL_Start(*req_ptr);

    spmv(alpha, A.on_proc, x, beta, b);

    MPIL_Wait(*req_ptr, MPI_STATUS_IGNORE);

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

int CG_persistent(ParMat& A, std::vector<double>& x, std::vector<double>& b)
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
    MPIL_Request* mpil_spmv_req;
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
    //int max_iter = ((int)(1.3*b.size())) + 2;
    int max_iter = 500;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm, &mpil_spmv_req);

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
        spmv(1.0, A, p, 0.0, Ap, mpil_comm, &mpil_spmv_req);
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
            spmv(-1.0, A, x, 1.0, r, mpil_comm, &mpil_spmv_req);
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

    MPIL_Request_free(&mpil_req);
    MPIL_Request_free(&mpil_spmv_req);
    MPIL_Info_free(&mpil_info);
    MPIL_Comm_free(&mpil_comm);

    return iter;
}

int CG(ParMat& A, std::vector<double>& x, std::vector<double>& b)
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
    //int max_iter = ((int)(1.3*b.size())) + 2;
    int max_iter = 500;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);

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
        spmv(1.0, A, p, 0.0, Ap, mpil_comm);
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
            spmv(-1.0, A, x, 1.0, r, mpil_comm);
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

    MPIL_Comm_free(&mpil_comm);

    return iter;
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    MPIL_Comm* mpil_comm;
    MPIL_Comm_init(&mpil_comm, MPI_COMM_WORLD);

    double t0, tfinal;
    int n_iters = 10;

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
    spmv(1.0, A, x, 0.0, b, mpil_comm);

    int conv_iter;
    std::vector<double> r;
    double sum;
    double norm_b;
    norm_b = 0;
    for (int i = 0; i < b.size(); i++)
        norm_b += b[i] * b[i];
    MPI_Allreduce(MPI_IN_PLACE, &norm_b, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    norm_b = sqrt(norm_b);

    /********* Test for Correctness  ***************/
    // PMPI
    MPIL_Set_allreduce_algorithm(ALLREDUCE_PMPI);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + PMPI: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Recursive Doubling
    MPIL_Set_allreduce_algorithm(ALLREDUCE_RECURSIVE_DOUBLING);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL RD: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);
    // Dissemination Node-Aware 
    MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_LOC);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL NA: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Dissemination Locality-Aware 
    MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_ML);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL LA: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Dissemination High-Radix 
    MPIL_Set_allreduce_algorithm(ALLREDUCE_DISSEMINATION_RADIX);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Radix: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Persistent Recursive Doubling
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RECURSIVE_DOUBLING);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG_persistent(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Persistent RD: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Persistent Dissemination Node-Aware 
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_DISSEMINATION_LOC);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG_persistent(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Persistent NA: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Persistent Dissemination Locality-Aware 
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_DISSEMINATION_ML);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG_persistent(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Persistent LA: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Persistent Dissemination High-Radix 
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_DISSEMINATION_RADIX);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG_persistent(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Persistent Radix: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Persistent RMA Hierarchical
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RMA_HIERARCHICAL);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG_persistent(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Persistent RMA Hier: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

        // Persistent RMA Hierarchical Early Bird
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RMA_HIERARCHICAL_EARLYBIRD);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG_persistent(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Persistent RMA Hier EarlyBird: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    // Persistent RMA Multileader
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RMA_MULTILEADER);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG_persistent(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Persistent RMA ML: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

        // Persistent RMA Multileader Early Bird
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RMA_MULTILEADER_EARLYBIRD);
    std::fill(x.begin(), x.end(), 0);
    conv_iter = CG_persistent(A, x, b);
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm);
    sum = 0;
    for (int i = 0; i < r.size(); i++)
        sum += r[i] * r[i];
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    if (rank == 0) printf("CG + MPIL Persistent RMA ML EarlyBird: %d iter, norm %e\n", 
            conv_iter, sqrt(sum) / norm_b);

    /***********************************************************/

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

    // Persistent Recursive Doubling
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RECURSIVE_DOUBLING);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG_persistent(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with Persistent MPIL Recursive Doubling Allreduce: %e\n", t0);

    // Persistent Dissemination Node-Aware
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_DISSEMINATION_LOC);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG_persistent(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with Persistent MPIL Node-Aware Dissemination Allreduce: %e\n", t0);

    // Persistent Dissemination Locality-Aware
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_DISSEMINATION_ML);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG_persistent(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with Persistent MPIL Locality-Aware Dissemination Allreduce: %e\n", t0);

    // Persistent Dissemination High-Radix
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_DISSEMINATION_RADIX);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG_persistent(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with Persistent MPIL High-Radix Dissemination Allreduce: %e\n", t0);

    // Persistent RMA Hierarchical
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RMA_HIERARCHICAL);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG_persistent(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with Persistent MPIL RMA Hierarchical Allreduce: %e\n", t0);

    // Persistent RMA Hierarchical Early Bird
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RMA_HIERARCHICAL_EARLYBIRD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG_persistent(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with Persistent MPIL RMA Hierarchical EarlyBird Allreduce: %e\n", t0);

    // Persistent RMA Multileader
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RMA_MULTILEADER);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG_persistent(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with Persistent MPIL RMA ML Allreduce: %e\n", t0);

    // Persistent RMA Multileader Early Bird
    MPIL_Set_allreduce_init_algorithm(ALLREDUCE_RMA_MULTILEADER_EARLYBIRD);
    t0 = MPI_Wtime();
    for (int i = 0; i < n_iters; i++)
    {
        std::fill(x.begin(), x.end(), 0);
        CG_persistent(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / n_iters;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("CG with Persistent MPIL RMA ML EarlyBird Allreduce: %e\n", t0);

    MPIL_Comm_free(&mpil_comm);

    MPI_Finalize();
}
