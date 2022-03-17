#include <iostream>
#include <vector>
#include <cstdlib>
#include "./include/sampler.h"
#include "./include/Parser.h"
#include <math.h>

using namespace std;

struct Args{
    std::string config_folder = "./input/santiago";
    std::string output_file = "./output/results.txt";
    std::string contact_mtx = "";
    int moving_t = -1;
    double R0 = 2.6;
    double r1 = 1.0;
    double r2 = 1.0;
    unsigned int maxT = 175;
    double c = 0.0;
};

void read_args(int argc, char* argv[], Args& args){
    for(int i=0;i<argc;i++){
        std::string act_param = argv[i];
        if(act_param=="--config"){
            args.config_folder = argv[++i];
            if(args.config_folder.back() != '/') args.config_folder += '/';
        }
        else if(act_param=="--out") args.output_file = argv[++i];
        else if(act_param=="--contact_mtx") args.contact_mtx = argv[++i];
        else if(act_param=="--moving_t") args.moving_t = std::stoi(argv[++i]);
        else if(act_param=="--R0") args.R0 = std::stod(argv[++i]);
        else if(act_param=="--r1") args.r1 = std::stod(argv[++i]);
        else if(act_param=="--r2") args.r2 = std::stod(argv[++i]);
        else if(act_param=="--maxT") args.maxT = std::stoi(argv[++i]);
        else if(act_param=="--c") args.c = std::stod(argv[++i]);
        //else if(act_param=="--verbose"){ args.verbose=true;++i;}
    }
}

double seasonality(double c, double t){
    return 0.5*c*cos(2*M_PI*t/366.0)+(1-0.5*c);
}

/**
 * Compute beta given R0
 */
double get_beta(double R0, double mu, double eigenV)
{
    double beta = R0 * mu / eigenV;
    return beta;
}


/**
 * Multiply matrix by scalar
 */
vector<vector<double>> scalarProductMat(vector<vector<double>> C, double r, int K)
{
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            C[i][j] = C[i][j] * r;
    return C;
}


/**
 * Sum two matrix with same dimension
 */
vector<vector<double>> sumMat(vector<vector<double>> C1, vector<vector<double>> C2, int K)
{
    vector<vector<double>> C(K, vector<double>(K, 0.0));
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j ++)
            C[i][j] = C1[i][j] + C2[i][j];
    return C;
}


/**
 * Get sigma_i for population i
 */
double get_sigma(int j, vector<vector<double>> &sigmas)
{
    double sigma = 0.0;
    for (int i = 0; i < sigmas[j].size(); i++)
        sigma += sigmas[j][i];
    return sigma;
}


/**
 * Get effective N for a specific age group
 */
double get_Nk_eff(int pop_idx, int age_idx, double tau, vector<vector<double>> &Nk, vector<vector<double>> &sigmas, vector<double> &sigmas_j)
{
    double Neff = Nk[pop_idx][age_idx] / (1 + sigmas_j[pop_idx] / tau);
    for (int i = 0; i < Nk.size(); i++)
        Neff += Nk[i][age_idx] / (1 + sigmas_j[i] / tau) * sigmas[i][pop_idx] / tau;
    return Neff;
}


/**
 * Get single lambda
 */
double get_lambda(int pop_idx, int age_idx, double tau, double beta, vector<vector<double>> &Nk_eff, vector<vector<double>> &I, vector<vector<double>> &C, vector<vector<double>> &sigmas, vector<double> &sigmas_j)
{
    double lambda = 0.0;
    for (int i = 0; i < I.size(); i++)
        if (i == pop_idx)
            for (int k = 0; k < I[0].size(); k++)
                lambda += (C[age_idx][k] / Nk_eff[pop_idx][k]) * (I[pop_idx][k] / (1 + sigmas_j[pop_idx] / tau));
        else
            for (int k = 0; k < I[0].size(); k++)
                lambda += (C[age_idx][k] / Nk_eff[pop_idx][k]) * (I[i][k] / (1 + sigmas_j[i] / tau)) * (sigmas[i][pop_idx] / tau);
    lambda = lambda * beta;
    return lambda;
}


/**
 * Get total lambda
 */
double get_lambda_tot(int pop_idx, int age_idx, double tau, double beta, vector<vector<double>> &Nk_eff, vector<vector<double>> &I, vector<vector<vector<double>>> &C, vector<vector<double>> &sigmas, vector<double> &sigmas_j)
{
    double lambda_tot = 0.0;
    double lambda_ji = 0.0;
    double lambda_jj = get_lambda(pop_idx, age_idx, tau, beta, Nk_eff, I, C[pop_idx], sigmas, sigmas_j);
    lambda_tot += lambda_jj / (1 + sigmas_j[pop_idx] / tau);

    for (int i = 0; i < I.size(); i++)
    {
        lambda_ji = get_lambda(i, age_idx, tau, beta, Nk_eff, I, C[i], sigmas, sigmas_j);
        lambda_tot += lambda_ji / (1 + sigmas_j[pop_idx] / tau) * (sigmas[pop_idx][i] / tau);
    }
    return lambda_tot;
}


void update_moving(
    int K,
    int Npop,
    vector<vector<double>>& Nk_eff,
    vector<vector<double>>& sigmas,
    vector<double>& sigmas_j,
    vector<vector<vector<double>>>& C,
    vector<vector<double>>& C1,
    vector<vector<double>>& C2,
    vector<double>& r){
    
}

/**
 * Main
 */
int main(int argc, char *argv[])
{
    // -- args and parser
    Args args;
    read_args(argc, argv, args);
    Parser parser = Parser(args.config_folder);

    // number of comunas, age groups, and simulations per parameters
    int Npop = parser.parse_Npop();
    // Number of age categories
    int K = 16;

    std::cout<<Npop<<std::endl;

    // parameters
    double mu = 1 / 2.5;
    double eps = 1 / 4.0;
    double tau = 3.0;
    double R0 = args.R0;
    double beta = get_beta(R0, mu, 16.204308331681283);
    vector<double> betas;


    // commuting
    vector<vector<double>> sigmas = parser.parse_commuting();
    vector<double> sigmas_j;
    for (int i = 0; i < Npop; i++)
        sigmas_j.push_back(get_sigma(i, sigmas));

    // compartments
    vector<vector<double>> S = parser.parse_compartments("S");
    vector<vector<double>> L = parser.parse_compartments("L");
    vector<vector<double>> I = parser.parse_compartments("I");
    vector<vector<double>> R = parser.parse_compartments("R");
    vector<vector<double>> Nk = parser.parse_compartments("N");

    // contacts (home, other)
    vector<vector<double>> C1 = parser.parse_contacts(args.contact_mtx);
    vector<vector<double>> C2 = parser.parse_contacts(2);

    vector<double> r1,r2;
    if(args.r1 == -1.0){
        r1 = parser.parse_r(1);
        r2 = parser.parse_r(2);
    }
    else{
        r1 = std::vector<double>(Npop, args.r1);
        r2 = std::vector<double>(Npop, args.r2);
    }

    vector<vector<vector<double>>> C;
    for (int i = 0; i < Npop; i++){
        //C.push_back(sumMat(C1, C2, K));
        C.push_back(sumMat(scalarProductMat(C1, r1[i], K), scalarProductMat(C2, r1[i], K), K));
    }

    std::cout << "Parsing complete..." << '\n';

    // compute N effective k
    vector<vector<double>> Nk_eff(Npop, vector<double>(K, 0.0));
    for (int i = 0; i < Npop; i++)
        for (int k = 0; k < K; k++)
            Nk_eff[i][k] = get_Nk_eff(i, k, tau, Nk, sigmas, sigmas_j);

    // next step data
    vector<vector<double>> Snext(Npop, vector<double>(K, 0.0));
    vector<vector<double>> Lnext(Npop, vector<double>(K, 0.0));
    vector<vector<double>> Inext(Npop, vector<double>(K, 0.0));
    vector<vector<double>> Rnext(Npop, vector<double>(K, 0.0));
    
    vector<vector<double>> I_new(Npop, vector<double>(K, 0.0));
    
    double newL = 0.0;
    double newI = 0.0;
    double newR = 0.0;
    double lambda = 0.0;

    // write results header
    ofstream resFile(args.output_file);
    for (int i = 0; i < Npop; i++)
        for (int k = 0; k < K; k++)
            resFile << "I_" << to_string(i) << "_" << to_string(k) << "," << "R_" << to_string(i) << "_" << to_string(k) << "," ;
    resFile << "\n";

    std::cout << "Start Simulation" << '\n';
    // simulate
    int T0 = 243;
    for (int t = T0; t < T0+args.maxT; t++)
    {
        std::cout<<"\r"<<t<<"    "<<std::flush;
        if (t == args.moving_t){ // restrictions
            C.clear();
            for (int i = 0; i < Npop; i++)
                C.push_back(sumMat(scalarProductMat(C1, r2[i], K), scalarProductMat(C2, r2[i], K), K));

            sigmas.clear();
            sigmas_j.clear();
            sigmas = parser.parse_commuting();
            for (int i = 0; i < Npop; i++)
                sigmas_j.push_back(get_sigma(i, sigmas));

            for (int i = 0; i < Npop; i++)
                for (int k = 0; k < K; k++)
                    Nk_eff[i][k] = get_Nk_eff(i, k, tau, Nk, sigmas, sigmas_j);
        }

        double act_beta = beta*seasonality(args.c, t);
        for (int i = 0; i < Npop; i++)
        {
            for (int k = 0; k < K; k++)
            {
                // S -> L
                lambda = get_lambda_tot(i, k, tau, act_beta, Nk_eff, I, C, sigmas, sigmas_j);
                newL = binomial(S[i][k], lambda);

                // L -> I
                newI = binomial(L[i][k], eps);
                I_new[i][k]=newI;

                // I -> R
                newR = binomial(I[i][k], mu);

                // update
                Snext[i][k] = S[i][k] - newL;
                Lnext[i][k] = L[i][k] + newL - newI;
                Inext[i][k] = I[i][k] + newI - newR;
                Rnext[i][k] = R[i][k] + newR;
            }
        }

        // update next step data and write results
        for (int i = 0; i < Npop; i++)
        {
            for (int k = 0; k < K; k++)
            {
                S[i][k] = Snext[i][k];
                L[i][k] = Lnext[i][k];
                I[i][k] = Inext[i][k];
                R[i][k] = Rnext[i][k];
                resFile << I_new[i][k] << "," << R[i][k] << ",";
                //resFile << I[i][k] << "," << R[i][k] << ",";
            }
        }
        resFile << "\n";
    }
    std::cout << "End of Simulation" << '\n';
    return 0;
}
