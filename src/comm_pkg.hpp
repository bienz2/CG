#ifndef MPI_SPARSE_MAT_HPP
#define MPI_SPARSE_MAT_HPP

#include <mpi.h>
#include <vector>
#include "sparse_mat.hpp"

struct CommData
{
    int n_msgs;
    int size_msgs;
    std::vector<int> procs;
    std::vector<int> ptr;
    std::vector<int> counts;
    std::vector<int> idx;
    std::vector<MPI_Request> req;
};

struct CommPkg
{
    Comm send_data;
    Comm recv_data;
}

void form_recv_comm(ParMat& A, CommData& recv_comm)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Gather first col for all processes into list
    std::vector<int> first_cols(num_procs + 1);
    MPI_Allgather(
        &A.first_col, 1, MPI_INT, first_cols.data(), 1, MPI_INT, MPI_COMM_WORLD);
    first_cols[num_procs] = A.global_cols;

    // Map Columns to Processes
    int proc      = 0;
    int prev_proc = -1;
    for (int i = 0; i < A.off_proc_num_cols; i++)
    {
        int global_col = A.off_proc_columns[i];
        while (first_cols[proc + 1] <= global_col)
        {
            proc++;
        }
        if (proc != prev_proc)
        {
            recv_comm.procs.push_back(proc);
            recv_comm.ptr.push_back((U)(i));
            prev_proc = proc;
        }
    }

    // Set Recv Sizes
    recv_comm.ptr.push_back((U)(A.off_proc_num_cols));
    recv_comm.n_msgs    = recv_comm.procs.size();
    recv_comm.size_msgs = A.off_proc_num_cols;
    if (recv_comm.n_msgs == 0)
    {
        return;
    }

    recv_comm.req.resize(recv_comm.n_msgs);
    recv_comm.counts.resize(recv_comm.n_msgs);
    for (int i = 0; i < recv_comm.n_msgs; i++)
    {
        recv_comm.counts[i] = recv_comm.ptr[i + 1] - recv_comm.ptr[i];
    }
}

// Must Form Recv Comm before Send!
void form_send_comm(ParMat& A)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::vector<long> recv_buf;
    std::vector<int> sizes(num_procs, 0);
    int proc, count, ctr;
    MPI_Status recv_status;

    // Allreduce to find size of data I will receive
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        sizes[A.recv_comm.procs[i]] = A.recv_comm.ptr[i + 1] - A.recv_comm.ptr[i];
    }
    MPI_Allreduce(
        MPI_IN_PLACE, sizes.data(), num_procs, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    A.send_comm.size_msgs = sizes[rank];

    // Send a message to every process that I will need data from
    // Tell them which global indices I need from them
    int msg_tag = 1234;
    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc = A.recv_comm.procs[i];
        MPI_Isend(&(A.off_proc_columns[A.recv_comm.ptr[i]]),
                  A.recv_comm.counts[i],
                  MPI_LONG,
                  proc,
                  msg_tag,
                  MPI_COMM_WORLD,
                  &(A.recv_comm.req[i]));
    }

    // Wait to receive values
    // until I have received fewer than the number of global indices I am waiting on
    if (A.send_comm.size_msgs)
    {
        A.send_comm.idx.resize(A.send_comm.size_msgs);
        recv_buf.resize(A.send_comm.size_msgs);
    }
    ctr = 0;
    A.send_comm.ptr.push_back(0);
    while (ctr < A.send_comm.size_msgs)
    {
        // Wait for a message
        MPI_Probe(MPI_ANY_SOURCE, msg_tag, MPI_COMM_WORLD, &recv_status);

        // Get the source process and message size
        proc = recv_status.MPI_SOURCE;
        A.send_comm.procs.push_back(proc);
        MPI_Get_count(&recv_status, MPI_LONG, &count);
        A.send_comm.counts.push_back(count);

        // Receive the message, and add local indices to send_comm
        MPI_Recv(&(recv_buf[ctr]),
                 count,
                 MPI_LONG,
                 proc,
                 msg_tag,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++)
        {
            A.send_comm.idx[ctr + i] = (recv_buf[ctr + i] - A.first_col);
        }
        ctr += count;
        A.send_comm.ptr.push_back((U)(ctr));
    }

    // Set send sizes
    A.send_comm.n_msgs = A.send_comm.procs.size();

    if (A.send_comm.n_msgs)
    {
        A.send_comm.req.resize(A.send_comm.n_msgs);
    }

    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }
}

template <typename U>
CommPkg& form_comm(ParMat& A)
{
    CommPkg comm_pkg;

    form_recv_comm(A);

    form_send_comm(A);
}

#endif
