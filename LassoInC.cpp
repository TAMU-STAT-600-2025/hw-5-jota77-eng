#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Soft-thresholding function, returns scalar
// [[Rcpp::export]]
double soft_c(double a, double lambda){if (a >  lambda) return a - lambda;
if (a < -lambda) return a + lambda;
return 0.0;
  // Your function code goes here
}

// Lasso objective function, returns scalar
// [[Rcpp::export]]
double lasso_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& beta, double lambda){arma::colvec r = Ytilde - Xtilde * beta;
  const double n = static_cast<double>(Xtilde.n_rows);
  return 0.5 * arma::dot(r, r) / n + lambda * arma::norm(beta, 1);
  // Your function code goes here
}

// Lasso coordinate-descent on standardized data with one lamdba. Returns a vector beta.
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, double lambda, const arma::colvec& beta_start, double eps = 0.001){const int n = static_cast<int>(Xtilde.n_rows);
  const int p = static_cast<int>(Xtilde.n_cols);
  const int max_iter = 10000; 
  
  arma::colvec beta = beta_start;
  if ((int)beta.n_elem != p) {
    beta.set_size(p);
    beta.zeros();
  }
  arma::rowvec xtx = arma::sum(arma::square(Xtilde), 0); // 1 x p
  arma::colvec r = Ytilde - Xtilde * beta;
  
  for (int it = 0; it < max_iter; ++it) {
    double max_delta = 0.0;
    
    for (int j = 0; j < p; ++j) {
      const double bj_old = beta[j];
      const double xtxj  = xtx[j];
      
      // rho_j = x_j^T (r + x_j * b_old)
      const arma::colvec xj = Xtilde.col(j);
      const double rho_j = arma::dot(xj, r) + xtxj * bj_old;
      
      // S(rho_j, n*lambda) / (x_j^T x_j)
      const double bj_new = soft_c(rho_j, static_cast<double>(n) * lambda) / xtxj;
      
      const double db = bj_new - bj_old;
      if (db != 0.0) {
        // r <- r - x_j * (b_new - b_old)
        r -= xj * db;
        beta[j] = bj_new;
        const double ad = std::abs(db);
        if (ad > max_delta) max_delta = ad;
      }
    }
    
    if (max_delta < eps) break;
  }
  
  return beta;
  // Your function code goes here
}  

// Lasso coordinate-descent on standardized data with supplied lambda_seq. 
// You can assume that the supplied lambda_seq is already sorted from largest to smallest, and has no negative values.
// Returns a matrix beta (p by number of lambdas in the sequence)
// [[Rcpp::export]]
arma::mat fitLASSOstandardized_seq_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& lambda_seq, double eps = 0.001){const int p = static_cast<int>(Xtilde.n_cols);
  const int L = static_cast<int>(lambda_seq.n_elem);
  
  arma::mat B(p, L, arma::fill::zeros);
  arma::colvec beta_start(p, arma::fill::zeros);
  
  for (int k = 0; k < L; ++k) {
    const double lam = lambda_seq[k];
    arma::colvec beta_k = fitLASSOstandardized_c(Xtilde, Ytilde, lam, beta_start, eps);
    B.col(k) = beta_k;
    beta_start = beta_k; // warm start
  }
  
  return B;
  // Your function code goes here
}