#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"

#include <vector>
using namespace std;

//
//  benchmarking program
//
int main(int argc, char **argv)
{
    int navg, nabsavg = 0, numthreads;
    double dmin, absmin = 1.0, davg, absavg = 0.0;

    if (find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

    particle_t *particles = (particle_t *)malloc(n * sizeof(particle_t));
    set_size(n);
    init_particles(n, particles);

    int dim = (int)ceil(size / cutoff);
    vector<int> **bin = new vector<int> *[dim];
    for (int i = 0; i < dim; i++)
    {
        bin[i] = new vector<int>[dim];
    }

    omp_lock_t **locks = new omp_lock_t *[dim];
    for (int i = 0; i < dim; i++)
    {
        locks[i] = new omp_lock_t[dim];
    }

    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            omp_init_lock(&locks[i][j]);
        }
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
    #pragma omp parallel private(dmin)
    {
        numthreads = omp_get_num_threads();
        for (int step = 0; step < NSTEPS; step++)
        {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            //
            //  binning
            //
            int row = 0;
            int col = 0;
            #pragma omp for
            for (int i = 0; i < n; i++)
            {
                row = (int)floor(particles[i].x / cutoff);
                col = (int)floor(particles[i].y / cutoff);
                omp_set_lock(&locks[row][col]);
                bin[row][col].push_back(i);
                omp_unset_lock(&locks[row][col]);
            }

            //
            //  compute all forces (bin by bin)
            //
            #pragma omp for collapse(2) reduction(+:navg) reduction(+:davg)
            for (int row = 0; row < dim; row++)
            {
                for (int col = 0; col < dim; col++)
                {
                    for (int d = 0; d < bin[row][col].size(); d++)
                    {
                        int i = bin[row][col][d];
                        particles[i].ax = particles[i].ay = 0;
                        for (int r = row - 1; r <= row + 1 && r < dim; r++)
                        {
                            if (r >= 0)
                            {
                                for (int c = col - 1; c <= col + 1 && c < dim; c++)
                                {
                                    if (c >= 0)
                                    {
                                        for (int b = 0; b < bin[r][c].size(); b++)
                                        {
                                            int j = bin[r][c][b];
                                            apply_force(particles[i], particles[j], &dmin, &davg, &navg);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            /*
            //
            // compute all forces (particle by particle)
            //
            #pragma omp for reduction(+:navg) reduction(+:davg)
            for (int i = 0; i < n; i++)
            {
                particles[i].ax = particles[i].ay = 0;
                int row = (int)floor(particles[i].x / cutoff);
                int col = (int)floor(particles[i].y / cutoff);
                for (int r = row - 1; r <= row + 1 && r < dim; r++)
                {
                    if (r >= 0)
                    {
                        for (int c = col - 1; c <= col + 1 && c < dim; c++)
                        {
                            if (c >= 0)
                            {
                                //printf("r:%d, c:%d\n", r, c);
                                for (int b = 0; b < bin[r][c].size(); b++)
                                {
                                    int j = bin[r][c][b];
                                    apply_force(particles[i], particles[j], &dmin, &davg, &navg);
                                }
                            }
                        }
                    }
                }
            }
            */

            #pragma omp for collapse(2)
            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    bin[i][j].clear();
                }
            }

            /*
            #pragma omp for reduction(+:navg) reduction(+:davg)
            for (int i = 0; i < n; i++)
            {
                particles[i].ax = particles[i].ay = 0;
                for (int j = 0; j < n; j++)
                    apply_force(particles[i], particles[j], &dmin, &davg, &navg);
            }
            */

            //
            //  move particles
            //
            #pragma omp for
            for (int i = 0; i < n; i++)
                move(particles[i]);

            if (find_option(argc, argv, "-no") == -1)
            {
                //
                //  compute statistical data
                //
                #pragma omp master
                if (navg)
                {
                    absavg += davg / navg;
                    nabsavg++;
                }

                #pragma omp critical
                if (dmin < absmin)
                    absmin = dmin;

                //
                //  save if necessary
                //
                #pragma omp master
                if (fsave && (step % SAVEFREQ) == 0)
                    save(fsave, n, particles);
            }
        }
    }

    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            omp_destroy_lock(&locks[i][j]);
        }
    }
    simulation_time = read_timer() - simulation_time;

    printf("n = %d,threads = %d, simulation time = %g seconds", n, numthreads, simulation_time);

    if (find_option(argc, argv, "-no") == -1)
    {
        if (nabsavg)
            absavg /= nabsavg;
        //
        //  -The minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf(", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4)
            printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8)
            printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if (fsum)
        fprintf(fsum, "%d %d %g\n", n, numthreads, simulation_time);

    //
    // Clearing space
    //

    for (int i = 0; i < dim; i++)
    {
        delete[] locks[i];
    }
    delete[] locks;

    for (int i = 0; i < dim; i++)
    {
        delete[] bin[i];
    }
    delete[] bin;

    if (fsum)
        fclose(fsum);

    free(particles);

    if (fsave)
        fclose(fsave);

    return 0;
}
