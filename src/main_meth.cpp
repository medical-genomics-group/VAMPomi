#include <mpi.h>
#include <iostream>
#include <numeric>
#include "utilities.hpp"
#include "data.hpp"
#include "vamp.hpp"

int main(int argc, char** argv)
{
    // starting parallel MPI processes
    int required_MPI_level = MPI_THREAD_MULTIPLE;
    int provided_MPI_level;
    int rank = 0; // current rank number
    int nranks = 0; //number of ranks
    MPI_Init_thread(NULL, NULL, required_MPI_level, &provided_MPI_level);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Command line options
    const Options opt(argc, argv);

    std::string out_dir = opt.get_out_dir();
    std::string out_name = opt.get_out_name();
    std::string model = opt.get_model();
    int verbosity = opt.get_verbosity();

    size_t Mt = opt.get_Mt(); // Total number of markers
    size_t N = opt.get_N(); // Number of samples
    int C = opt.get_C(); // Number of covariates

    // Divide work for parallel ranks
    std::vector<double> MS = divide_work(Mt);
    int M = MS[0]; // Numbe of markers for current rank
    int S = MS[1]; // Offset of part of markers processed by current rank 
    int Mm = MS[2]; // Index of marker max

    // Infer model parameters and calculate test R2
    if (opt.get_run_mode() == "infere"){

        // Get command line options
        std::string phenfp = opt.get_phen_file(); // Phenotype file
        std::string mrkfp = opt.get_meth_file();
        double alpha_scale = opt.get_alpha_scale();

        // Reading train set
        data dataset(phenfp, mrkfp, model, N , M, Mt, S, rank, alpha_scale);

        // Reading test set
        const std::string mrkfp_test = opt.get_meth_file_test();
        const std::string phenfp_test = opt.get_phen_file_test(); // currently it is only supported passing one pheno files as an input argument

        size_t N_test = opt.get_N_test();
        size_t Mt_test = opt.get_Mt_test();

        data dataset_test(phenfp_test, mrkfp_test, model, N_test, M, Mt_test, S, rank, alpha_scale);

        // Initialize model hyperparameters
        double h2 = opt.get_h2(); // heritability
        double gamw = 1.0 / (1.0 - h2); // Noise precision
        double gam1 = opt.get_gam1(); // Signal noise precision
        
        int max_iter = opt.get_iterations();
        int learn_vars = opt.get_learn_vars();
        int CG_max_iter = opt.get_CG_max_iter();
        int use_lmmse_damp = opt.get_use_lmmse_damp();
        double rho = opt.get_rho();
        std::vector<double> vars = opt.get_vars();
        std::vector<double> probs = opt.get_probs();

        // Read true signals, if provided
        std::vector<double> true_signal;
        if(opt.get_true_signal_file() != "")
            true_signal = mpi_read_vec_from_file(opt.get_true_signal_file(), M, S);
        else 
            true_signal = std::vector<double> (M, 0.0);

        // ---------------- running VAMP algorithm -------------------- //
        vamp emvamp(N, 
                    M,
                    Mt,
                    C,
                    gam1,
                    gamw,
                    max_iter, 
                    CG_max_iter,
                    use_lmmse_damp,
                    rho,
                    learn_vars,
                    vars,
                    probs,
                    true_signal,
                    out_dir,
                    out_name,
                    model,
                    verbosity,
                    rank); // Create VAMP instance
        
        std::vector<double> x_est = emvamp.infere(&dataset); // Infer model parameters

        // x_hat estimates normalization
        for (int i0 = 0; i0 < x_est.size(); i0++)
            x_est[i0] *= sqrt( (double) N_test );

        std::vector<double> z_test = dataset_test.Ax(x_est.data()); // predicted outcome
        std::vector<double> y_test = dataset_test.get_phen(); //true outcome

        // Calculate L2 prediction error
        double l2_pred_err2 = 0;
        for (int i0 = 0; i0 < N_test; i0++){
            l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
        }  

        double stdev = calc_stdev(y_test); // standard deviation of true outcome y
        double r2 = 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ); // R2 test

        // Print metrics
        if (rank == 0){
            std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
            std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
            std::cout << "test R2 = " << r2 << std::endl;
        }
    }
    else if (opt.get_run_mode() == "test") // just analyzing the result on the test data
    {
       
        // reading test set
        const std::string mrkfp_test = opt.get_meth_file_test();
        const std::string pheno_test = opt.get_phen_file_test(); // currently it is only supported passing one pheno files as an input argument

        int N_test = opt.get_N_test();

        std::vector<double> MS = divide_work(Mt);
        int M = MS[0];
        int S = MS[1];

        double alpha_scale = opt.get_alpha_scale();


        data dataset_test(pheno_test, mrkfp_test, model, N_test, M, Mt, S, rank, alpha_scale);
        
        std::vector<double> y_test = dataset_test.get_phen();

        // Open output file for storing test metrics (R2)
        MPI_File outcsv_test_fh;
        std::string outcsv_test_fp = opt.get_out_dir() + "/" + opt.get_out_name() + ".csv";
        MPI_File_delete(outcsv_test_fp.c_str(), MPI_INFO_NULL);
        check_mpi(MPI_File_open(MPI_COMM_WORLD,
                            outcsv_test_fp.c_str(),
                            MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
                            MPI_INFO_NULL,
                            &outcsv_test_fh),
                            __LINE__, __FILE__);

        // parse estimate file name
        std::string est_file_name = opt.get_estimate_file();
        int pos_dot = est_file_name.find(".");
        std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
        if (rank == 0)
            std::cout << "est_file_name = " << est_file_name << std::endl;

        int pos_it = est_file_name.rfind("it");
        std::vector<int> iter_range = opt.get_test_iter_range();
        int min_it = iter_range[0];
        int max_it = iter_range[1];
        if (rank == 0)
            std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

        for (int it = min_it; it <= max_it; it++){
            std::vector<double> params;
            std::vector<double> x_est;
            std::string est_file_name_it = est_file_name.substr(0, pos_it) + "it_" + std::to_string(it) + "." + end_est_file_name;

            if (end_est_file_name == "bin")
                x_est = mpi_read_vec_from_file(est_file_name_it, M, S);
            else
                x_est = read_vec_from_file(est_file_name_it, M, S);

            // normalization of estimates
            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N_test );
                
            // Predictions
            std::vector<double> z_test = dataset_test.Ax(x_est.data());

            // L2 prediction error
            double l2_pred_err2 = 0;
            for (int i0 = 0; i0 < N_test; i0++){
                l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
            }  

            // Standard deviation of true outcome
            double stdev = calc_stdev(y_test);
            double r2 = 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ); // R2 test
            if (rank == 0){
                std::cout << r2 << ", ";
                params.push_back(r2);
            }
            if (rank == 0) {
                write_ofile_csv(outcsv_test_fh, it, &params);
            }
        }

        
    }
    else if (opt.get_run_mode() == "predict"){

        std::string phenfp = opt.get_phen_file();
        double alpha_scale = opt.get_alpha_scale();

        data dataset(phenfp, opt.get_meth_file(), model, N, M, Mt, S, rank, alpha_scale);
        
        std::vector<double> y = dataset.get_phen();

        std::vector<double> beta_true;
        if(opt.get_true_signal_file() != "" )
            beta_true = read_vec_from_file(opt.get_true_signal_file(), M, S);

        std::string est_file_name = opt.get_estimate_file();

        if (rank == 0)
            std::cout << "est_file_name = " << est_file_name << std::endl;

        std::vector<double> x_est;
        x_est = mpi_read_vec_from_file(est_file_name, M, S);

        for (int i0 = 0; i0 < x_est.size(); i0++){
            x_est[i0] *= sqrt( (double) N );
            beta_true[i0] *= sqrt( (double) N );
        }
                
        std::vector<double> z = dataset.Ax(x_est.data());

        double l2_pred_err2 = 0;
        for (int i0 = 0; i0 < N; i0++){
            l2_pred_err2 += (y[i0] - z[i0]) * (y[i0] - z[i0]);
        }  

        double stdev = calc_stdev(y);
        if (rank == 0){
            std::cout << "R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y.size() ) << std::endl;
        }
        
        // Saving predictions
        std::string filepath_out_z = opt.get_out_dir() + opt.get_out_name() + "_z.yest";
        store_vec_to_file(filepath_out_z, z);
        if (rank == 0)
            std::cout << "prediction filepath is " << filepath_out_z << std::endl;
        }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}