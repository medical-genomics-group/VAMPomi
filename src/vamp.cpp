#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <numeric> // contains std::accumulate
#include <random>
#include <omp.h>
#include <fstream>
#include <cfloat>
#include "vamp.hpp"
#include "data.hpp"
#include "vamp_probit.cpp"
#include "utilities.hpp"

// VAMP constructor 
vamp::vamp( int N,
            int M,
            int Mt, 
            int C, 
            double gam1, 
            double gamw, 
            int max_iter,
            int CG_max_iter,
            int use_lmmse_damp,
            double rho,
            int learn_vars,
            std::vector<double> vars,
            std::vector<double> probs, 
            std::vector<double> true_signal, 
            std::string out_dir, 
            std::string out_name, 
            std::string model,
            int verbosity,
            int rank):

            N(N),
            M(M),
            Mt(Mt),
            C(C),
            gam1(gam1),
            gamw(gamw),
            max_iter(max_iter),
            CG_max_iter(CG_max_iter),
            use_lmmse_damp(use_lmmse_damp),
            rho(rho),
            learn_vars(learn_vars),
            vars(vars),
            probs(probs),
            true_signal(true_signal),
            out_dir(out_dir),
            out_name(out_name),
            model(model),
            verbosity(verbosity),
            rank(rank){

                x1_hat = std::vector<double> (M, 0.0);
                x2_hat = std::vector<double> (M, 0.0);
                r1 = std::vector<double> (M, 0.0);
                r2 = std::vector<double> (M, 0.0);
                p1 = std::vector<double> (N, 0.0);
    
                MPI_Comm_size(MPI_COMM_WORLD, &nranks);

                // we scale mixture variances of effects by N (since the design matrix is scaled by 1/sqrt(N))
                for (int i = 0; i < vars.size(); i++)
                    this->vars[i] *= N;

                setup_io();
}

// --------------- VAMP - MAIN INFERENCE PROCEDURE ----------------
std::vector<double> vamp::infere( data* dataset ){

    y = (*dataset).get_phen(); // True outcome
    
    // deciding between linear regression and probit regression
    if (!strcmp(model.c_str(), "linear"))
        return infere_linear(dataset);
    else if (!strcmp(model.c_str(), "bin_class"))
        return infere_bin_class(dataset);
    else
        throw "Invalid model specification!";

    return std::vector<double> (M, 0.0);
}

// ------------ VAMP linear model ------------- 
std::vector<double> vamp::infere_linear(data* dataset){

    std::vector<double> metrics(4,0); // Error metrics from denoising and lmmse steps to save to csv [R2, x1_corr | R2, x2_corr]
    std::vector<double> params(5,0); // Parameters from denoising and lmmse steps to save to csv [alpha1 | gam1 | alpha2 | gam2 | gamw ]

    std::vector<double> x1_hat_d(M, 0.0);
    std::vector<double> x1_hat_d_prev(M, 0.0);
    std::vector<double> x1_hat_scaled(M, 0.0);
    std::vector<double> x1_hat_prev(M, 0.0);

    std::vector<double> y =  (*dataset).get_phen();

    r1 = std::vector<double> (M, 0.0);   

    // initializing z1_hat and p2
    z1_hat = std::vector<double> (N, 0.0);
    p2 = std::vector<double> (N, 0.0);

    std::vector< std::vector<double> > Z = (*dataset).get_covs();
    if (C > 0){
        cov_eff = std::vector<double> (C, 0.0);
    }

    std::vector<double> gg;
    double sqrtN = sqrt(N);

    // ---------------- starting VAMP iterations ----------------
    for (int it = 1; it <= max_iter; it++)
    {    
        if (rank == 0)
            std::cout << std::endl << "********************" << std::endl << "iteration = "<< it << std::endl << "********************" << std::endl;
        
        // --------------- Covariate effects -----------------
        double start_covar = MPI_Wtime(); // measure time for covariates
        if (it == 1 && C > 0){
            gg = z1_hat;
            cov_eff = Newton_method_cov(y, gg, Z, cov_eff);
            if (rank == 0){
                for (int i0=0; i0<C; i0++){
                    std::cout << "cov_eff[" << i0 << "] = " << cov_eff[i0] << ", ";
                    if (i0 % 4 == 3)
                        std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            for (int i=0; i<N; i++){
                y[i] -= inner_prod(Z[i], cov_eff, 0);
            }
        }

        double stop_covar = MPI_Wtime();

        if (rank == 0)
            std::cout << "time for covariates effects update = " << stop_covar - start_covar << " seconds." << std::endl;

        // --------------- Denoising step -----------------
        double start_denoising = MPI_Wtime();

        if (rank == 0)
            std::cout << "->DENOISING" << std::endl;

        // updating parameters of prior distribution
        probs_before = probs;
        vars_before = vars;
        
        x1_hat_prev = x1_hat;

        // keeping value of Onsager from a previous iteration
        double alpha1_prev = alpha1;

        // re-estimating the error variance
        double gam1_reEst_prev;
        int it_revar = 1;

        for (; it_revar <= auto_var_max_iter; it_revar++){

            // new signal estimate
            for (int i = 0; i < M; i++)
                x1_hat[i] = g1(r1[i], gam1);

            std::vector<double> x1_hat_m_r1 = x1_hat;

            for (int i0 = 0; i0 < x1_hat_m_r1.size(); i0++)
                x1_hat_m_r1[i0] = x1_hat_m_r1[i0] - r1[i0];

            // new MMSE estimate
            double sum_d = 0;
            for (int i=0; i<M; i++)
            {
                x1_hat_d[i] = g1d(r1[i], gam1);
                sum_d += x1_hat_d[i];
            }

            alpha1 = 0;
            MPI_Allreduce(&sum_d, &alpha1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            alpha1 /= Mt;
            eta1 = gam1 / alpha1;

            if (it <= 1)
                break;

            // because we want both EM updates to be performed by maximizing likelihood
            // with respect to the old gamma

            gam1_reEst_prev = gam1;
            gam1 = std::min( std::max(  1.0 / (1.0/eta1 + l2_norm2(x1_hat_m_r1, 1)/Mt), gamma_min ), gamma_max );

            updatePrior();

            if(verbosity == 1)
                if (rank == 0 && it_revar % 1 == 0)
                    std::cout << "[old] it_revar = " << it_revar << ": gam1 = " << gam1 << std::endl;

            if ( abs(gam1 - gam1_reEst_prev) < 1e-3 )
                break;
        }

        // saving gam1 estimates
        //gam1s.push_back(gam1);

        if (rank == 0)
            std::cout << "A total of " << std::max(it_revar - 1,1) << " variance and prior tuning iterations were performed" << std::endl;
        
        // damping 
        if (it > 1){ 
            std::vector<double> x1_hat_temp = x1_hat;
            for (int i = 0; i < M; i++)
                x1_hat[i] = rho * x1_hat[i] + (1-rho) * x1_hat_prev[i];
            alpha1 = rho * alpha1 + (1-rho) * alpha1_prev;
        }
            
        z1 = (*dataset).Ax(x1_hat.data());

        // saving x1_hat
        std::string filepath_out = out_dir + out_name + "_it_" + std::to_string(it) + ".bin";
        int S = (*dataset).get_S();
        for (int i0=0; i0<x1_hat_scaled.size(); i0++)
            x1_hat_scaled[i0] =  x1_hat[i0] / sqrtN;
        mpi_store_vec_to_file(filepath_out, x1_hat_scaled, S, M);

        if (rank == 0)
           std::cout << "x1_hat filepath_out is " << filepath_out << std::endl;

        std::string filepath_out_r1 = out_dir + out_name + "_r1_it_" + std::to_string(it) + ".bin";
        std::vector<double> r1_scaled = r1;
        for (int i0=0; i0<r1_scaled.size(); i0++)
            r1_scaled[i0] =  r1[i0] / sqrtN;
        mpi_store_vec_to_file(filepath_out_r1, r1_scaled, S, M);

        if (rank == 0)
           std::cout << "r1_hat filepath_out is " << filepath_out_r1 << std::endl;

        gam_before = gam2;
        gam2 = std::min(std::max(eta1 - gam1, gamma_min), gamma_max);
        //if (rank == 0){
        //    std::cout << "eta1 = " << eta1 << std::endl;
        //    std::cout << "gam2 = " << gam2 << std::endl;
        //}

        r2_prev = r2;

        for (int i = 0; i < M; i++)
            r2[i] = (eta1 * x1_hat[i] - gam1 * r1[i]) / gam2;

        if (use_lmmse_damp == 1){
            double xi = std::min(2*rho, 1.0);
            if (it > 1){
                gam2 = 1.0 / pow( xi / sqrt(gam2) + (1-xi) / sqrt(gam_before), 2);
            } 
        }

        // if the true value of the signal is known, we print out the true gam2
        double se_dev = 0;
        for (int i0=0; i0<M; i0++){
            se_dev += (r2[i0] - sqrtN*true_signal[i0])*(r2[i0] - sqrtN*true_signal[i0]);
        }
        double se_dev_total = 0;
        MPI_Allreduce(&se_dev, &se_dev_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double gam2_true = Mt / se_dev_total;
        //if (rank == 0)
        //    std::cout << "true gam2 = " << Mt / se_dev_total << std::endl;


        // new place for prior update
        //if (auto_var_max_iter == 0 || it <=2)
        //    updatePrior();

        err_measures(dataset, 1, metrics);

        // Save and print hyperparameters
        params[0] = alpha1;
        params[1] = gam1;

        if(rank == 0){
            std::cout << "alpha1 = " << alpha1 << std::endl;
            std::cout << "gam1 = " << gam1 << std::endl;
            std::cout << "gam2 = " << gam2 << std::endl;
            std::cout << "true gam2 = " << gam2_true << std::endl;
        }

        double end_denoising = MPI_Wtime();

        // ----------------- LMMSE --------------------

        double start_lmmse_step = MPI_Wtime();

        if (rank == 0)
            std::cout << "______________________" << std::endl<< "->LMMSE" << std::endl;

        // running conjugate gradient solver to compute LMMSE
        double start_CG = MPI_Wtime();

        std::vector<double> v;

        v = (*dataset).ATx(y.data());

        for (int i = 0; i < M; i++)
            v[i] = gamw * v[i] + gam2 * r2[i];

        if (it == 1)
            x2_hat = precondCG_solver(v, std::vector<double>(M, 0.0), gamw, 1, dataset); // precond_change!
        else
            x2_hat = precondCG_solver(v, mu_CG_last, gamw, 1, dataset); // precond_change!

        double end_CG = MPI_Wtime();

        if (rank == 0)
            std::cout << "CG took "  << end_CG - start_CG << " seconds." << std::endl;


        double start_onsager = MPI_Wtime();

        alpha2 = g2d_onsager(gam2, gamw, dataset);

        double end_onsager = MPI_Wtime();

        if (rank == 0)
            std::cout << "onsager took "  << end_onsager - start_onsager << " seconds." << std::endl;
        
        //if (rank == 0)
        //    std::cout << "alpha2 = " << alpha2 << std::endl;
        
        eta2 = gam2 / alpha2;

        // re-estimating gam2 <- new
        std::vector<double> x2_hat_m_r2 = x2_hat;
        for (int i0 = 0; i0 < x2_hat_m_r2.size(); i0++)
            x2_hat_m_r2[i0] = x2_hat_m_r2[i0] - r2[i0];

        if (auto_var_max_iter >= 1 && it > 2){
            gam2 = std::min( std::max(  1 / (1/eta2 + l2_norm2(x2_hat_m_r2, 1)/Mt), gamma_min ), gamma_max );
        }
        //gam2s.push_back(gam2);

        if (rank == 0)
            std::cout << "gam2 re-est = " << gam2 << std::endl;

        gam1 = std::min( std::max( eta2 - gam2, gamma_min ), gamma_max );
        
        //if (rank == 0)
        //    std::cout << "gam1 = " << gam1 << std::endl;

        for (int i = 0; i < M; i++)
            r1[i] = (eta2 * x2_hat[i] - gam2 * r2[i]) / gam1;

        // if the true value of the signal is known, we print out the true gam1
        double se_dev1 = 0;
        for (int i0=0; i0<M; i0++)
            se_dev1 += (r1[i0]- sqrtN*true_signal[i0])*(r1[i0]- sqrtN*true_signal[i0]);
        double se_dev_total1 = 0;
        MPI_Allreduce(&se_dev1, &se_dev_total1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double gam1_true = Mt / se_dev_total1;
        //if (rank == 0)
        //    std::cout << "true gam1 = " << Mt / se_dev_total1 << std::endl; 

        // learning a noise precision parameter
        updateNoisePrec(dataset);
   
        // printing out error measures
        err_measures(dataset, 2, metrics);

        // Save and print hyperparameters
        params[2] = alpha2;
        params[3] = gam2;
        params[4] = gamw;

        if(rank == 0){
            std::cout << "alpha2 = " << alpha2 << std::endl;
            std::cout << "gam2 = " << gam2 << std::endl;
            std::cout << "gam1 = " << gam1 << std::endl;
            std::cout << "true gam1 = " << gam1_true << std::endl;
            std::cout << "gamw = " << gamw << std::endl;
        }

        // Store mixture probabilities and variances
        std::vector<double> prior_params;
        prior_params.push_back(probs.size());
        for(int i=0; i < probs.size(); i++)
            prior_params.push_back(probs[i]);
        for(int i=0; i < vars.size(); i++)
            prior_params.push_back(vars[i] / (double) N);

        if (rank == 0){
            std::cout << "...storing parameters to CSV files" << std::endl;
            write_ofile_csv(out_params_fh, it, &params);
            write_ofile_csv(out_metrics_fh, it, &metrics);
            write_ofile_csv(out_prior_fh, it, &prior_params);
        }
        
        double end_lmmse_step = MPI_Wtime();
        total_comp_time += end_denoising - start_denoising + end_lmmse_step - start_lmmse_step;

        if (rank == 0){
            std::cout << "LMMSE step took "  << end_lmmse_step - start_lmmse_step << " seconds." << std::endl;
            std::cout << "Total iteration time = " << end_denoising - start_denoising + end_lmmse_step - start_lmmse_step << std::endl;
            std::cout << "Total computation time so far = " << total_comp_time << std::endl;
            std::cout << std::endl << std::endl;
        }
    }

    check_mpi(MPI_File_close(&out_params_fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&out_metrics_fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&out_prior_fh), __LINE__, __FILE__);

    MPI_Barrier(MPI_COMM_WORLD);

    // returning scaled version of the effects
    return x1_hat_scaled;          
}

double vamp::g1(double y, double gam1) { 
    
    double sigma = 1 / gam1;
    double eta_max = *(std::max_element(vars.begin(), vars.end()));
    double pk = 0, pkd = 0, val;

    if (sigma < 1e-10 && sigma > -1e-10) {
        return y;
    }

    for (int i = 0; i < probs.size(); i++){

        double expe_sum =  - 0.5 * pow(y,2) * ( eta_max - vars[i] ) / ( vars[i] + sigma ) / ( eta_max + sigma );
        double z = probs[i] / sqrt( vars[i] + sigma ) * exp( expe_sum );

        pk = pk + z;
        z = z / ( vars[i] + sigma ) * y;
        pkd = pkd - z; 
    }

    val = (y + sigma * pkd / pk);
    
    return val;
}

double vamp::g1d(double y, double gam1) { 
    
        double sigma = 1 / gam1;
        double eta_max = *std::max_element(vars.begin(), vars.end());
        double pk = 0, pkd = 0, pkdd = 0;
        
        if (sigma < 1e-10 && sigma > -1e-10) {
            return 1;
        }

        for (int i = 0; i < probs.size(); i++){

            double expe_sum = - 0.5 * pow(y,2) * ( eta_max - vars[i] ) / ( vars[i] + sigma ) / ( eta_max + sigma );
            double z = probs[i] / sqrt( vars[i] + sigma ) * exp( expe_sum );

            pk = pk + z;
            z = z / ( vars[i] + sigma ) * y;
            pkd = pkd - z;

            double z2 = z / ( vars[i] + sigma ) * y;
            pkdd = pkdd - probs[i] / pow( vars[i] + sigma, 1.5 ) * exp( expe_sum ) + z2;
            
        } 

        double val = (1 + sigma * ( pkdd / pk - pow( pkd / pk, 2) ) );

        return val;
}

double vamp::g2d_onsager(double gam2, double tau, data* dataset) { // shared between linear and binary classification model
    
    std::random_device rd;
    std::bernoulli_distribution bern(0.5);

    bern_vec = std::vector<double> (M, 0.0);

    for (int i = 0; i < M; i++)
        bern_vec[i] = (2*bern(rd) - 1) / sqrt(Mt); // Bernoulli variables are sampled independently

    invQ_bern_vec = precondCG_solver(bern_vec, tau, 0, dataset); // precond_change

    double onsager = gam2 * inner_prod(bern_vec, invQ_bern_vec, 1); // because we want to calculate gam2 * Tr[(gamw * X^TX + gam2 * I)^(-1)] / Mt

    return onsager;    
}


void vamp::updateNoisePrec(data* dataset){

    y = (*dataset).get_phen();  

    std::vector<double> temp = (*dataset).Ax(x2_hat.data());

    for (int i = 0; i < N; i++)  // because length(y) = N
        temp[i] -= y[i];
    
    double temp_norm2 = l2_norm2(temp, 0); 
    
    std::vector<double> trace_corr_vec_N;
    std::vector<double> trace_corr_vec_M;

    trace_corr_vec_N = (*dataset).Ax(invQ_bern_vec.data());
    trace_corr_vec_M = (*dataset).ATx(trace_corr_vec_N.data());
    
    double trace_corr = inner_prod(bern_vec, trace_corr_vec_M, 1) * Mt; // because we took u ~ Bern({-1,1} / sqrt(Mt), 1/2)

    if (rank == 0){
        std::cout << "l2_norm2(temp) / N = " << temp_norm2 / N << std::endl;
        std::cout << "trace_correction / N = " << trace_corr / N << std::endl;
    }

    gamw = (double) N / (temp_norm2 + trace_corr);
}

void vamp::updatePrior() {
    
        double noise_var = 1 / gam1;
        double lambda = 1 - probs[0];

        std::vector<double> omegas = probs;
        for (int j = 1; j < omegas.size(); j++) // omegas is of length L
            omegas[j] /= lambda;
                       
        // calculating normalized beta and pin
        int it;

        for (it = 0; it < EM_max_iter; it++){

            double max_sigma = *std::max_element(vars.begin(), vars.end()); // std::max_element returns iterators, not values

            std::vector<double> probs_prev = probs;
            std::vector<double> vars_prev = vars;
            std::vector< std::vector<double> > gammas;
            std::vector< std::vector<double> > beta;
            std::vector<double> pin(M, 0.0);
            std::vector<double> v;

            for (int i = 0; i < M; i++){

                std::vector<double> temp; // of length (L-1)
                std::vector<double> temp_gammas;

                for (int j = 1; j < probs.size(); j++ ){

                    double num = lambda * omegas[j] * exp( - pow(r1[i], 2) / 2 * (max_sigma - vars[j]) / (vars[j] + noise_var) / (max_sigma + noise_var) ) / sqrt(vars[j] + noise_var) / sqrt(2 * M_PI);
                    double num_gammas = gam1 * r1[i] / ( 1 / vars[j] + gam1 );   

                    temp.push_back(num);
                    temp_gammas.push_back(num_gammas);
                }

                double sum_of_elems = std::accumulate(temp.begin(), temp.end(), decltype(temp)::value_type(0));
            
                for (int j = 0; j < temp.size(); j++ )
                    temp[j] /= sum_of_elems;
                
                beta.push_back(temp);
                gammas.push_back(temp_gammas);
                
                pin[i] = 1 / ( 1 + (1-lambda) / sqrt(2 * M_PI * noise_var) * exp( - pow(r1[i], 2) / 2 * max_sigma / noise_var / (noise_var + max_sigma) ) / sum_of_elems );
            } 

            for (int j = 1; j < probs.size(); j++)
                v.push_back( 1.0 / ( 1.0 / vars[j] + gam1 ) ); // v is of size (L-1) in the end
            
            lambda = accumulate(pin.begin(), pin.end(), 0.0); // / pin.size();

            double lambda_total = 0;

            MPI_Allreduce(&lambda, &lambda_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            lambda = lambda_total / Mt;
        
            for (int i = 0; i < M; i++){
                for (int j = 0; j < (beta[0]).size(); j++ ){
                    gammas[i][j] = beta[i][j] * ( gammas[i][j] * gammas[i][j] + v[j] );
                }
            }

            //double sum_of_pin = std::accumulate(pin.begin(), pin.end(), decltype(pin)::value_type(0));
            double sum_of_pin = lambda_total;

            for (int j = 0; j < (beta[0]).size(); j++){ // of length (L-1)
                double res = 0, res_gammas = 0;
                for (int i = 0; i < M; i++){
                    res += beta[i][j] * pin[i];
                    res_gammas += gammas[i][j] * pin[i];
                }

                double res_gammas_total = 0;
                double res_total = 0;
                MPI_Allreduce(&res_gammas, &res_gammas_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&res, &res_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                if (learn_vars == 1)
                    vars[j+1] = res_gammas_total / res_total;
                omegas[j+1] = res_total / sum_of_pin;
                probs[j+1] = lambda * omegas[j+1];

            }

            probs[0] = 1 - lambda;
        
            double distance_probs = 0, norm_probs = 0;
            double distance_vars = 0, norm_vars = 0;

            for (int j = 0; j < probs.size(); j++){

                distance_probs += ( probs[j] - probs_prev[j] ) * ( probs[j] - probs_prev[j] );
                norm_probs += probs[j] * probs[j];
                distance_vars += ( vars[j] - vars_prev[j] ) * ( vars[j] - vars_prev[j] );
                norm_vars += vars[j] * vars[j];
            }
            double dist_probs = sqrt(distance_probs / norm_probs);
            double dist_vars = sqrt(distance_vars / norm_vars);

            if (verbosity == 1)
                if (rank == 0)
                    std::cout << "it = " << it << ": dist_probs = " << dist_probs << " & dist_vars = " << dist_vars << std::endl;
            if ( dist_probs < EM_err_thr  && dist_vars < EM_err_thr )
                break;   
        }
    
        if (verbosity == 1)
            if (rank == 0)  
                std::cout << "Final number of prior EM iterations = " << std::min(it + 1, EM_max_iter) << " / " << EM_max_iter << std::endl;
}

std::vector<double> vamp::lmmse_mult(std::vector<double> v, double tau, data* dataset){ // multiplying with (tau*A^TAv + gam2*v)

    if (v == std::vector<double>(M, 0.0))
        return std::vector<double>(M, 0.0);

    std::vector<double> res(M, 0.0);
    size_t phen_size = N;
    std::vector<double> res_temp(phen_size, 0.0);
    res_temp = (*dataset).Ax(v.data());
    res = (*dataset).ATx(res_temp.data());

    for (int i = 0; i < M; i++){
        res[i] *= tau;
        res[i] += gam2 * v[i];
    }

    return res;
}

std::vector<double> vamp::precondCG_solver(std::vector<double> v, double tau, int denoiser, data* dataset){

    // we start with approximation x0 = 0
    std::vector<double> mu(M, 0.0);
    return precondCG_solver(v, mu, tau, denoiser, dataset);
}

std::vector<double> vamp::precondCG_solver(std::vector<double> v, std::vector<double> mu_start, double tau, int denoiser, data* dataset){

    // preconditioning part
    std::vector<double> diag(M, 1.0);

    for (int j=0; j<M; j++)
        diag[j] = tau * (N-1) / N + gam2;
         
    std::vector<double> mu = mu_start;
    std::vector<double> d;
    std::vector<double> r = lmmse_mult(mu, tau, dataset);

    for (int i0=0; i0<M; i0++)
        r[i0] = v[i0] - r[i0];

    std::vector<double> z(M, 0.0);

    std::transform (r.begin(), r.end(), diag.begin(), z.begin(), std::divides<double>());

    std::vector<double> p = z;
    std::vector<double> Apalpha(M, 0.0);
    std::vector<double> palpha(M, 0.0);

    double alpha, beta;
    double prev_onsager = 0;

    for (int i = 0; i < CG_max_iter; i++){

        // d = A*p
        d = lmmse_mult(p, tau, dataset);
        alpha = inner_prod(r, z, 1) / inner_prod(d, p, 1);
        
        for (int j = 0; j < M; j++)
            palpha[j] = alpha * p[j];

        std::transform (mu.begin(), mu.end(), palpha.begin(), mu.begin(), std::plus<double>());

        if (denoiser == 0){

            double onsager = gam2 * inner_prod(v, mu, 1);
            double rel_err;

            if (onsager != 0)
                rel_err = abs( (onsager - prev_onsager) / onsager ); 
            else
                rel_err = 1;

            if (rel_err < 1e-8)
                break;

            prev_onsager = onsager;

            if (rank == 0 && verbosity == 1)
                std::cout << "[CG onsager] it = " << i << ": relative error for onsager is " << rel_err << std::endl;

        }

        for (int j = 0; j < p.size(); j++)
            Apalpha[j] = d[j] * alpha;

        beta = pow(inner_prod(r, z, 1), -1);

        std::transform (r.begin(), r.end(), Apalpha.begin(), r.begin(), std::minus<double>());
        std::transform (r.begin(), r.end(), diag.begin(), z.begin(), std::divides<double>());

        beta *= inner_prod(r, z, 1);

        for (int j = 0; j < p.size(); j++)
            p[j] = z[j] + beta * p[j];

        // stopping criteria
        double norm_v = sqrt(l2_norm2(v, 1));
        double norm_z = sqrt(l2_norm2(z, 1));
        double rel_err = sqrt( l2_norm2(r, 1) ) / norm_v;
        double norm_mu = sqrt( l2_norm2(mu, 1) );
        double err_tol = 1e-5;

        if (rank == 0 && verbosity == 1)
            std::cout << "[CG] it = " << i << ": ||r_it|| / ||RHS|| = " << rel_err << ", ||x_it|| = " << norm_mu << ", ||z|| / ||RHS|| = " << norm_z /  norm_v << std::endl;

        if (rel_err < err_tol) 
            break;
    }
    if (denoiser == 1)
        mu_CG_last = mu;

    return mu;
 }


void vamp::err_measures(data *dataset, int ind, std::vector<double>& metrics){

    double scale = 1.0 / (double) N;
    
    // correlation
    if (ind == 1){

        double corr = inner_prod(x1_hat, true_signal, 1) / sqrt( l2_norm2(x1_hat, 1) * l2_norm2(true_signal, 1) );
        metrics[1] = corr;
        if ( rank == 0 )
            std::cout << "Corr(x1_hat, x0) = " << corr << std::endl;  

        double l2_norm2_x1_hat = l2_norm2(x1_hat, 1);
        double l2_norm2_true_signal = l2_norm2(true_signal, 1);

    }
    else if (ind == 2){

        double corr_2 = inner_prod(x2_hat, true_signal, 1) / sqrt( l2_norm2(x2_hat, 1) * l2_norm2(true_signal, 1) );
        metrics[3] = corr_2;
        if (rank == 0)
            std::cout << "Corr(x2_hat, x0)= " << corr_2 << std::endl;

        double l2_norm2_x2_hat = l2_norm2(x2_hat, 1);
        double l2_norm2_true_signal = l2_norm2(true_signal, 1);
        
    }
    

    // l2 signal error
    std::vector<double> temp(M, 0.0);
    double l2_norm2_xhat;

    if (ind == 1){
        for (int i = 0; i< M; i++)
            temp[i] = sqrt(scale) * x1_hat[i] - true_signal[i];

        l2_norm2_xhat = l2_norm2(x1_hat, 1) * scale;
    }
    else if (ind == 2){
        for (int i = 0; i< M; i++)
            temp[i] = sqrt(scale) * x2_hat[i] - true_signal[i];

        l2_norm2_xhat = l2_norm2(x2_hat, 1) * scale;
    }


    double l2_signal_err = sqrt( l2_norm2(temp, 1) / l2_norm2(true_signal, 1) );
    if (rank == 0)
        std::cout << "L2(x_hat, x0)= " << l2_signal_err << std::endl;
    
    
    // l2 prediction error
    size_t phen_size = N;
    std::vector<double> tempNest(phen_size, 0.0);
    std::vector<double> tempNtrue(phen_size, 0.0);

    y = (*dataset).get_phen();

    std::vector<double> Axest;
    if (ind == 1)
        if (z1.size() > 0)
            Axest = z1;
        else
            Axest = (*dataset).Ax(x1_hat.data());
    else if (ind == 2)
    Axest = (*dataset).Ax(x2_hat.data());

    for (int i = 0; i < N; i++){ // N because length(y) = N
        tempNest[i] = -Axest[i] + y[i];
    }

    double l2_pred_err = sqrt(l2_norm2(tempNest, 0) / l2_norm2(y, 0));
    double R2 = 1 - l2_pred_err * l2_pred_err;

    if(ind == 1){
        metrics[0] = R2;
    }
    else if (ind == 2){
        metrics[2] = R2;
    }

    if (rank == 0){
        std::cout << "R2 = " << R2 << std::endl;
        std::cout << "L2(y_hat, y) = " << l2_pred_err << std::endl;
    }
    
    // prior distribution parameters
    if (rank == 0){
        std::cout << "Prior variances = ";

        for (int i = 0; i < vars.size(); i++)
            std::cout << vars[i] / (double) N << ' ';

        std::cout << std::endl;
        
        std::cout << "Prior probabilities = ";

        for (int i = 0; i < probs.size(); i++)
            std::cout << probs[i] << ' ';

        std::cout << std::endl;
        
        //std::cout << "gamw = " << gamw << std::endl;
    }
}

void vamp::setup_io(){

    std::string out_metrics_fp = out_dir + "/" + out_name + "_metrics.csv";
    MPI_File_delete(out_metrics_fp.c_str(), MPI_INFO_NULL);
    check_mpi(  MPI_File_open(MPI_COMM_WORLD,
                out_metrics_fp.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
                MPI_INFO_NULL,
                &out_metrics_fh),
                __LINE__, __FILE__);

    std::string out_params_fp = out_dir + "/" + out_name + "_params.csv";
    MPI_File_delete(out_params_fp.c_str(), MPI_INFO_NULL);
    check_mpi(  MPI_File_open(MPI_COMM_WORLD,
                out_params_fp.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
                MPI_INFO_NULL,
                &out_params_fh),
                __LINE__, __FILE__);

    std::string out_prior_fp = out_dir + "/" + out_name + "_prior.csv";
    MPI_File_delete(out_prior_fp.c_str(), MPI_INFO_NULL);
    check_mpi(  MPI_File_open(MPI_COMM_WORLD,
                out_prior_fp.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
                MPI_INFO_NULL,
                &out_prior_fh),
                __LINE__, __FILE__);
}