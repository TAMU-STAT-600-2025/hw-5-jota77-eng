#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// ---------------- soft-threshold ----------------
// [[Rcpp::export]]
double soft_c(double a, double lambda) {
  if (a >  lambda) return a - lambda;
  if (a < -lambda) return a + lambda;
  return 0.0;
}

// ---------------- objective ---------------------
// [[Rcpp::export]]
double lasso_c(const arma::mat& Xtilde,
               const arma::colvec& Ytilde,
               const arma::colvec& beta,
               double lambda) {
  arma::colvec r = Ytilde - Xtilde * beta;
  const double n = static_cast<double>(Xtilde.n_rows);
  return 0.5 * arma::dot(r, r) / n + lambda * arma::norm(beta, 1);
}

// --------- CD (standardized), single lambda -----
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_c(const arma::mat& Xtilde,
                                    const arma::colvec& Ytilde,
                                    double lambda,
                                    const arma::colvec& beta_start,
                                    double eps = 0.001) {
  const int p = static_cast<int>(Xtilde.n_cols);
  const int max_iter = 10000;
  
  arma::colvec beta = beta_start;
  if ((int)beta.n_elem != p) {
    beta.set_size(p);
    beta.zeros();
  }
  
  const double n = static_cast<double>(Xtilde.n_rows);
  arma::colvec xtx = arma::sum(arma::square(Xtilde), 0).t();  // p x 1
  arma::colvec r   = Ytilde - Xtilde * beta;                  // residual
  
  // objective: (1/(2n))||r||^2 + lambda ||beta||_1
  auto obj = [&](const arma::colvec& rr, const arma::colvec& b)->double {
    return 0.5 * arma::dot(rr, rr) / n + lambda * arma::norm(b, 1);
  };
  
  double f_prev = obj(r, beta);
  
  for (int it = 0; it < max_iter; ++it) {
    // one full coordinate sweep
    for (int j = 0; j < p; ++j) {
      const arma::colvec xj = Xtilde.col(j);
      const double bj_old   = beta[j];
      
      // z_j = (1/n) * sum( xj * (r + xj * bj_old) )    
      double acc = 0.0;
      const arma::uword nrows = Xtilde.n_rows;
      for (arma::uword i = 0; i < nrows; ++i) {
        const double xi = xj[i];
        acc += xi * (r[i] + xi * bj_old);
      }
      const double zj = acc / n;
      
      
      // b_j_new = soft(z_j, lambda)                    
      const double bj_new = soft_c(zj, lambda);
      
      // db = bj_new - bj_old; r -= xj * db; beta[j] = bj_new;
      
      
      
      const double db = bj_new - beta[j];
      if (db != 0.0) {
        r   -= xj * db;  // incremental residual update
        beta[j] = bj_new;
      }
    }
    
    // stop when objective decrease < eps (match R)
    const double f_new = obj(r, beta);
    if ((f_prev - f_new) < eps) break;
    f_prev = f_new;
  }
  
  return beta;
}

// ---- CD (standardized), user-supplied lambda seq ----
// [[Rcpp::export]]
arma::mat fitLASSOstandardized_seq_c(const arma::mat& Xtilde,
                                     const arma::colvec& Ytilde,
                                     const arma::colvec& lambda_seq,
                                     double eps = 0.001) {
  const int p = static_cast<int>(Xtilde.n_cols);
  const int L = static_cast<int>(lambda_seq.n_elem);
  
  arma::mat B(p, L, arma::fill::zeros);
  arma::colvec beta_start(p, arma::fill::zeros);
  
  for (int k = 0; k < L; ++k) {
    const double lam = lambda_seq[k];
    arma::colvec beta_k = fitLASSOstandardized_c(Xtilde, Ytilde, lam, beta_start, eps);
    B.col(k) = beta_k;
    beta_start = beta_k;  // warm start
  }
  return B;
}
