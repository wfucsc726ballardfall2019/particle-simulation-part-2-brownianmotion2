#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ )
        partition_offsets[i] = min( i * particle_per_proc, n );
    
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ )
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];


    double size = sqrt( 0.0005 * n );
    
    int bucket_dim = ceil(size/(0.01));
    std::vector<int> bucket_to_particle[bucket_dim][bucket_dim];
    std::vector<int> new_bucket_to_particle[bucket_dim][bucket_dim];
    std::vector<int> particle_to_bucket[n];

    n_proc = floor(sqrt(n_proc))*floor(sqrt(n_proc));
    int proc_dim = sqrt(n_proc);
    int bucket_per_proc = ceil(bucket_dim / sqrt(n_proc));
    int *x_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    int *x_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ ){
        x_offsets[i] = min( (i % proc_dim)*bucket_per_proc, bucket_dim-1 );
        if(i % proc_dim == proc_dim -1){
          x_sizes[i] = bucket_dim - (proc_dim-1) * bucket_per_proc;
        }
        else{
          x_sizes[i] = bucket_per_proc;
        }
    }
    

    int *y_offsets = (int*) malloc( (n_proc) * sizeof(int) );
    int *y_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ ){
        y_offsets[i] = min( floor(i/sqrt(n_proc)) * bucket_per_proc, bucket_dim-1 );
        if( i >= proc_dim* (proc_dim-1)){
          y_sizes[i] = bucket_dim - (proc_dim-1)*bucket_per_proc;
        }
        else{
          y_sizes[i] = bucket_per_proc;
        }
}
    
    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
    
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );
    MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );
    
    //initial fill the buckets
    MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );

    for (int i = 0; i < n; i++)
    {
      particles[i].ax = particles[i].ay = 0;
      int xbucket = floor(particles[i].x / (0.01));
      int ybucket = floor(particles[i].y / (0.01));
      bucket_to_particle[xbucket][ybucket].push_back(i);
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
    for (int step = 0; step < NSTEPS; step++)
    {
      navg = 0;
      dmin = 1.0;
      davg = 0.0;
      double temp = read_timer();
      //
      //  collect all global data locally (not good idea to do)
      //
      //MPI_Allgatherv(local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD);


//apply forces to our part
  
    for( int xbucket = x_offsets[rank]; xbucket < x_offsets[rank]+x_sizes[rank]; xbucket ++){
     for( int ybucket = y_offsets[rank]; ybucket < y_offsets[rank]+y_sizes[rank]; ybucket ++){
       //printf("xbucket %d, ybucket %d, proc. %d\n",xbucket,ybucket,rank);
      for(int k = 0; k < bucket_to_particle[xbucket][ybucket].size(); k++){
        
        int i = bucket_to_particle[xbucket][ybucket].at(k);
        int xmin,xmax,ymin,ymax;
        if(xbucket > 0){xmin = -1;}
                    else{ xmin = 0;}
                    if(xbucket < bucket_dim-1){ xmax = 1;}
                    else{ xmax = 0;}

                    if(ybucket > 0){ ymin = -1;}
                    else{ymin = 0;}
                    if(ybucket < bucket_dim-1){ ymax = 1;}
                    else{ ymax = 0;}

                    for( int nbucketx = xmin; nbucketx <= xmax; nbucketx++){
                        for( int nbuckety = ymin; nbuckety <= ymax; nbuckety++){
                            for(int l = 0; l < bucket_to_particle[xbucket+nbucketx][ybucket+nbuckety].size(); l++){
                              apply_force(particles[i], particles[bucket_to_particle[xbucket + nbucketx][ybucket + nbuckety].at(l)], &dmin, &davg, &navg);
                            }
                        }
                    }
      }
     }
    }
    

    // printf("1: %f\n",read_timer()-temp);
    // temp = read_timer();

     for( int xbucket = x_offsets[rank]; xbucket < x_offsets[rank]+x_sizes[rank]; xbucket ++){
     for( int ybucket = y_offsets[rank]; ybucket < y_offsets[rank]+y_sizes[rank]; ybucket ++){
      for(int k = 0; k < bucket_to_particle[xbucket][ybucket].size(); k++){
        
        int i = bucket_to_particle[xbucket][ybucket].at(k);
        move(particles[i]);
        
      }}}

    // printf("2: %f\n",read_timer()-temp);
    // temp = read_timer();


    for (int xbucket = 0; xbucket < bucket_dim; xbucket++)
    {
      for (int ybucket = 0; ybucket < bucket_dim; ybucket++)
      {
        new_bucket_to_particle[xbucket][ybucket].clear();
      }
    } 

// printf("3: %f\n",read_timer()-temp);
//     temp = read_timer();


  
  
    for(int sender_rank = 0; sender_rank<n_proc; sender_rank ++){
            int ybucket = y_offsets[sender_rank];
            for(int xbucket = x_offsets[sender_rank];  xbucket < x_offsets[sender_rank] + x_sizes[sender_rank]; xbucket++){
              
              int num_to_send = bucket_to_particle[xbucket][ybucket].size();
              MPI_Bcast(&num_to_send,1,MPI_INT,sender_rank,MPI_COMM_WORLD);
              for(int k = 0; k < num_to_send; k++){
                int pid;
                if(rank == sender_rank)
                pid = bucket_to_particle[xbucket][ybucket].at(k);
                MPI_Bcast(&pid,1,MPI_INT,sender_rank,MPI_COMM_WORLD);
              MPI_Bcast(&particles[pid],1,PARTICLE,sender_rank,MPI_COMM_WORLD);
              int new_xbucket = floor(particles[pid].x / (0.01));
                int new_ybucket = floor(particles[pid].y / (0.01));
                if(rank != sender_rank)
                new_bucket_to_particle[new_xbucket][new_ybucket].push_back(pid);

              }
            }
            

             ybucket = y_offsets[sender_rank]+y_sizes[sender_rank]-1;
            for(int xbucket = x_offsets[sender_rank];  xbucket < x_offsets[sender_rank] + x_sizes[sender_rank]; xbucket++){
              
              int num_to_send = bucket_to_particle[xbucket][ybucket].size();
              MPI_Bcast(&num_to_send,1,MPI_INT,sender_rank,MPI_COMM_WORLD);
              for(int k = 0; k < num_to_send; k++){
                int pid;
                if(rank == sender_rank)
                pid = bucket_to_particle[xbucket][ybucket].at(k);
                MPI_Bcast(&pid,1,MPI_INT,sender_rank,MPI_COMM_WORLD);
              MPI_Bcast(&particles[pid],1,PARTICLE,sender_rank,MPI_COMM_WORLD);
              int new_xbucket = floor(particles[pid].x / (0.01));
                int new_ybucket = floor(particles[pid].y / (0.01));
                if(rank != sender_rank)
                new_bucket_to_particle[new_xbucket][new_ybucket].push_back(pid);

              }
            }

            int xbucket = x_offsets[sender_rank];
            for(int ybucket = y_offsets[sender_rank]+1;  ybucket < y_offsets[sender_rank] + y_sizes[sender_rank]-1; ybucket++){
              
              int num_to_send = bucket_to_particle[xbucket][ybucket].size();
              MPI_Bcast(&num_to_send,1,MPI_INT,sender_rank,MPI_COMM_WORLD);
              for(int k = 0; k < num_to_send; k++){
                int pid;
                if(rank == sender_rank)
                pid = bucket_to_particle[xbucket][ybucket].at(k);
                MPI_Bcast(&pid,1,MPI_INT,sender_rank,MPI_COMM_WORLD);
              MPI_Bcast(&particles[pid],1,PARTICLE,sender_rank,MPI_COMM_WORLD);
              int new_xbucket = floor(particles[pid].x / (0.01));
                int new_ybucket = floor(particles[pid].y / (0.01));
                if(rank != sender_rank)
                new_bucket_to_particle[new_xbucket][new_ybucket].push_back(pid);

              }
            }

             xbucket = x_offsets[sender_rank]+x_sizes[sender_rank]-1;

            for(int ybucket = y_offsets[sender_rank]+1;  ybucket < y_offsets[sender_rank] + y_sizes[sender_rank]-1; ybucket++){
              
              int num_to_send = bucket_to_particle[xbucket][ybucket].size();
              MPI_Bcast(&num_to_send,1,MPI_INT,sender_rank,MPI_COMM_WORLD);
              for(int k = 0; k < num_to_send; k++){
                int pid;
                if(rank == sender_rank)
                pid = bucket_to_particle[xbucket][ybucket].at(k);
                MPI_Bcast(&pid,1,MPI_INT,sender_rank,MPI_COMM_WORLD);
              MPI_Bcast(&particles[pid],1,PARTICLE,sender_rank,MPI_COMM_WORLD);
              int new_xbucket = floor(particles[pid].x / (0.01));
                int new_ybucket = floor(particles[pid].y / (0.01));
                if(rank != sender_rank)
                new_bucket_to_particle[new_xbucket][new_ybucket].push_back(pid);

              }
            }
          }
 
        
// printf("4: %f\n",read_timer()-temp);
//     temp = read_timer();


for(int xbucket = x_offsets[rank];  xbucket < x_offsets[rank] + x_sizes[rank]; xbucket++){
  for(int ybucket = y_offsets[rank]; ybucket < y_offsets[rank] + y_sizes[rank]; ybucket++){
    for (int k = 0; k < bucket_to_particle[xbucket][ybucket].size(); k++){
      int i = bucket_to_particle[xbucket][ybucket].at(k);
      particles[i].ax = particles[i].ay = 0;
      int new_xbucket = floor(particles[i].x / (0.01));
      int new_ybucket = floor(particles[i].y / (0.01));
      new_bucket_to_particle[new_xbucket][new_ybucket].push_back(i);
    }
  }
}

// printf("5: %f\n",read_timer()-temp);
//     temp = read_timer();



for(int xbucket = 0;  xbucket < bucket_dim; xbucket++){
  for(int ybucket = 0; ybucket < bucket_dim; ybucket++){
    
    bucket_to_particle[xbucket][ybucket].clear();
  }
}

// printf("6: %f\n",read_timer()-temp);
//     temp = read_timer();


for(int xbucket = max(0,x_offsets[rank]-1);  xbucket < min(bucket_dim,x_offsets[rank]+x_sizes[rank]+1); xbucket++){
  for(int ybucket = max(0,y_offsets[rank]-1); ybucket < min(bucket_dim,y_offsets[rank]+y_sizes[rank]+1); ybucket++){
//    for (int k = 0; k < new_bucket_to_particle[xbucket][ybucket].size(); k++){
    bucket_to_particle[xbucket][ybucket]=new_bucket_to_particle[xbucket][ybucket];
  //}
}
}

// printf("7: %f\n",read_timer()-temp);
//     temp = read_timer();


    //
    //  save current step if necessary (slightly different semantics than in other codes)
    //
    if (find_option(argc, argv, "-no") == -1)
      if (fsave && (step % SAVEFREQ) == 0)
        save(fsave, n, particles);

    //
    //  compute all forces
    //

    if (find_option(argc, argv, "-no") == -1)
    {

      MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

      if (rank == 0)
      {
        //
        // Computing statistical data
        //
        if (rnavg)
        {
          absavg += rdavg / rnavg;
          nabsavg++;
        }
        if (rdmin < absmin)
          absmin = rdmin;
      }
      }

      //
      //  move particles
      //
    }
  simulation_time = read_timer() - simulation_time;

  if (rank == 0)
  {
    printf("n = %d, simulation time = %g seconds", n, simulation_time);

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
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
