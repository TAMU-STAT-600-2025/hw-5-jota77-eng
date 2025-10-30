# [ToDo] Standardize X and Y: center both X and Y; scale centered X
# X - n x p matrix of covariates
# Y - n x 1 response vector
standardizeXY <- function(X, Y){
  X <- as.matrix(X); Y <- as.numeric(Y)
  n <- nrow(X); p <- ncol(X)
  if(length(Y) != n) stop("X and Y have incompatible n.")
  
  # [ToDo] Center Y
  Ymean <- mean(Y)
  Ytilde <- Y - Ymean
  
  # [ToDo] Center and scale X
  Xmeans <- colMeans(X)
  Xc <- sweep(X, 2, Xmeans, FUN = "-")
  
  # weights: sqrt((X_j^T X_j) / n) after centering, before scaling
  weights <- sqrt(colSums(Xc^2) / n)
  # avoid division by zero for constant columns
  weights[weights == 0] <- 1
  
  # Scale X so that (1/n) * Xtilde_j^T Xtilde_j = 1
  Xtilde <- sweep(Xc, 2, weights, FUN = "/")
  
  # Return:
  # Xtilde - centered and appropriately scaled X
  # Ytilde - centered Y
  # Ymean - the mean of original Y
  # Xmeans - means of columns of X (vector)
  # weights - defined as sqrt(X_j^{\top}X_j/n) after centering of X but before scaling
  return(list(Xtilde = Xtilde, Ytilde = Ytilde, Ymean = Ymean, Xmeans = Xmeans, weights = weights))
}

# [ToDo] Soft-thresholding of a scalar a at level lambda 
# [OK to have vector version as long as works correctly on scalar; will only test on scalars]
soft <- function(a, lambda){
  sign(a) * pmax(abs(a) - lambda, 0)
}

# [ToDo] Calculate objective function of lasso given current values of Xtilde, Ytilde, beta and lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba - tuning parameter
# beta - value of beta at which to evaluate the function
lasso <- function(Xtilde, Ytilde, beta, lambda){
  n <- nrow(Xtilde)
  r <- as.numeric(Ytilde - Xtilde %*% beta)
  (sum(r^2) / (2*n)) + lambda * sum(abs(beta))
}

# [ToDo] Fit LASSO on standardized data for a given lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1 (vector)
# lamdba - tuning parameter
# beta_start - p vector, an optional starting point for coordinate-descent algorithm
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized <- function(Xtilde, Ytilde, lambda, beta_start = NULL, eps = 0.001){
  Xtilde <- as.matrix(Xtilde); Ytilde <- as.numeric(Ytilde)
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  #[ToDo]  Check that n is the same between Xtilde and Ytilde
  if(length(Ytilde) != n) stop("Xtilde and Ytilde have incompatible n.")
  
  #[ToDo]  Check that lambda is non-negative
  if(lambda < 0) stop("lambda must be non-negative.")
  
  #[ToDo]  Check for starting point beta_start. 
  # If none supplied, initialize with a vector of zeros.
  # If supplied, check for compatibility with Xtilde in terms of p
  if(is.null(beta_start)){
    beta <- rep(0, p)
  } else {
    if(length(beta_start) != p) stop("beta_start has wrong length.")
    beta <- as.numeric(beta_start)
  }
  
  #[ToDo]  Coordinate-descent implementation. 
  # Stop when the difference between objective functions is less than eps for the first time.
  # For example, if you have 3 iterations with objectives 3, 1, 0.99999,
  # your should return fmin = 0.99999, and not have another iteration
  # initialize residual and objective
  r <- as.numeric(Ytilde - Xtilde %*% beta)
  f_prev <- lasso(Xtilde, Ytilde, beta, lambda)
  
  repeat {
    # one full coordinate sweep
    for(j in 1:p){
      xj <- Xtilde[, j]
      bj_old <- beta[j]
      # z_j = (1/n) x_j^T (r + x_j * b_j)
      zj <- sum(xj * (r + xj * bj_old)) / n
      bj_new <- soft(zj, lambda)
      if(bj_new != bj_old){
        beta[j] <- bj_new
        r <- r - xj * (bj_new - bj_old)
      }
    }
    f_new <- (sum(r^2)/(2*n)) + lambda * sum(abs(beta))
    if((f_prev - f_new) < eps){  # first time under eps → stop
      break
    }
    f_prev <- f_new
  }
  fmin <- f_new
  # Return 
  # beta - the solution (a vector)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta, fmin = fmin))
}

# [ToDo] Fit LASSO on standardized data for a sequence of lambda values. Sequential version of a previous function.
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence,
#             is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized_seq <- function(Xtilde, Ytilde, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Check that n is the same between Xtilde and Ytilde
  Xtilde <- as.matrix(Xtilde); Ytilde <- as.numeric(Ytilde)
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if(length(Ytilde) != n) stop("Xtilde and Ytilde have incompatible n.")
  if(!is.null(lambda_seq)){
    lambda_seq <- sort(unique(lambda_seq[lambda_seq >= 0]), decreasing = TRUE)
    if(length(lambda_seq) == 0){
      warning("All supplied lambda values were invalid. Proceeding to compute default sequence.")
      lambda_seq <- NULL
    }
  }
 
  # [ToDo] Check for the user-supplied lambda-seq (see below)
  # If lambda_seq is supplied, only keep values that are >= 0,
  # and make sure the values are sorted from largest to smallest.
  # If none of the supplied values satisfy the requirement,
  # print the warning message and proceed as if the values were not supplied.
  
  
  # If lambda_seq is not supplied, calculate lambda_max 
  # (the minimal value of lambda that gives zero solution),
  # and create a sequence of length n_lambda as
  if(is.null(lambda_seq)){
    # lambda_max = max_j | (1/n) X_j^T Y |
    xTy <- as.numeric(crossprod(Xtilde, Ytilde)) / n
    lambda_max <- max(abs(xTy))
    lambda_seq = exp(seq(log(lambda_max), log(0.01), length = n_lambda))
  }
  
  
  # [ToDo] Apply fitLASSOstandardized going from largest to smallest lambda 
  # (make sure supplied eps is carried over). 
  # Use warm starts strategy discussed in class for setting the starting values.
  L <- length(lambda_seq)
  beta_mat <- matrix(0, nrow = p, ncol = L)
  fmin_vec <- rep(NA_real_, L)
  
  beta_start <- rep(0, p)
  for(i in 1:L){
    lam <- lambda_seq[i]
    fit <- fitLASSOstandardized(Xtilde, Ytilde, lam, beta_start = beta_start, eps = eps)
    beta_mat[, i] <- fit$beta
    fmin_vec[i] <- fit$fmin
    beta_start <- fit$beta  # warm start
  }
  
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value
  # fmin_vec - length(lambda_seq) vector of corresponding objective function values at solution
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, fmin_vec = fmin_vec))
}

# [ToDo] Fit LASSO on original data using a sequence of lambda values
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Center and standardize X,Y based on standardizeXY function
  std <- standardizeXY(X, Y)
  Xtilde <- std$Xtilde; Ytilde <- std$Ytilde
  weights <- std$weights; Xmeans <- std$Xmeans; Ymean <- std$Ymean
  
  # [ToDo] Fit Lasso on a sequence of values using fitLASSOstandardized_seq
  # (make sure the parameters carry over)
  ans <- fitLASSOstandardized_seq(Xtilde, Ytilde, lambda_seq = lambda_seq, n_lambda = n_lambda, eps = eps)
  lambda_seq <- ans$lambda_seq
  beta_std <- ans$beta_mat
 
  # [ToDo] Perform back scaling and centering to get original intercept and coefficient vector
  # for each lambda
  p <- ncol(X); L <- length(lambda_seq)
  beta_mat <- matrix(0, nrow = p, ncol = L)
  beta0_vec <- rep(NA_real_, L)
  for(i in 1:L){
    beta_orig <- beta_std[, i] / weights
    beta0 <- Ymean - sum(Xmeans * beta_orig)
    beta_mat[, i] <- beta_orig
    beta0_vec[i] <- beta0
  }
  
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, beta0_vec = beta0_vec))
}


# [ToDo] Fit LASSO and perform cross-validation to select the best fit
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# k - number of folds for k-fold cross-validation, default is 5
# fold_ids - (optional) vector of length n specifying the folds assignment (from 1 to max(folds_ids)), if supplied the value of k is ignored 
# eps - precision level for convergence assessment, default 0.001
cvLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, k = 5, fold_ids = NULL, eps = 0.001){
  # [ToDo] Fit Lasso on original data using fitLASSO
  X <- as.matrix(X); Y <- as.numeric(Y)
  n <- nrow(X)
  if(length(Y) != n) stop("X and Y have incompatible n.")
  if(is.null(fold_ids)){
    set.seed(1L)  
    fold_ids <- sample(rep(1:k, length.out = n))
  } else {
    k <- max(fold_ids)
  }
  # Fit on full data to fix lambda_seq
  full_fit <- fitLASSO(X, Y, lambda_seq = lambda_seq, n_lambda = n_lambda, eps = eps)
  lambda_seq <- full_fit$lambda_seq
  L <- length(lambda_seq)
  # [ToDo] If fold_ids is NULL, split the data randomly into k folds.
  # If fold_ids is not NULL, split the data according to supplied fold_ids.
  # Fit on full data to fix lambda_seq
  # Fit on full data to fix lambda_seq
  mse_mat <- matrix(NA_real_, nrow = k, ncol = L)
  
  for(f in 1:k){
    idx_val <- which(fold_ids == f)
    idx_tr  <- setdiff(seq_len(n), idx_val)
    
    Xtr <- X[idx_tr, , drop = FALSE]; Ytr <- Y[idx_tr]
    Xva <- X[idx_val, , drop = FALSE]; Yva <- Y[idx_val]
    
    fit_tr <- fitLASSO(Xtr, Ytr, lambda_seq = lambda_seq, eps = eps)
    beta0  <- fit_tr$beta0_vec            
    beta   <- fit_tr$beta_mat 

  # [ToDo] Calculate LASSO on each fold using fitLASSO,
  # and perform any additional calculations needed for CV(lambda) and SE_CV(lambda)
    XB <- Xva %*% beta                       # n_val × L
    Yhat_mat <- XB + matrix(1, nrow = nrow(Xva), ncol = 1) %*% t(beta0)  
    
    mse_mat[f, ] <- colMeans((Yva - Yhat_mat)^2)
  }
  
  cvm  <- colMeans(mse_mat)
  cvsd <- apply(mse_mat, 2, sd)
  cvse <- cvsd / sqrt(k)
  # [ToDo] Find lambda_min

  # [ToDo] Find lambda_1SE
  i_min <- which.min(cvm)
  lambda_min <- lambda_seq[i_min]
  thresh <- cvm[i_min] + cvse[i_min]
  i_1se <- min(which(cvm <= thresh))  
  lambda_1se <- lambda_seq[i_1se]
  
  
  # Return output
  # Output from fitLASSO on the whole data
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  # fold_ids - used splitting into folds from 1 to k (either as supplied or as generated in the beginning)
  # lambda_min - selected lambda based on minimal rule
  # lambda_1se - selected lambda based on 1SE rule
  # cvm - values of CV(lambda) for each lambda
  # cvse - values of SE_CV(lambda) for each lambda
  return(list(lambda_seq = lambda_seq,
              beta_mat   = full_fit$beta_mat,
              beta0_vec  = full_fit$beta0_vec,
              fold_ids   = fold_ids,
              lambda_min = lambda_min,
              lambda_1se = lambda_1se,
              cvm        = cvm,
              cvse       = cvse))
}

