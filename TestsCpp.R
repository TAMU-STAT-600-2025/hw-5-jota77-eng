
# Header for Rcpp and RcppArmadillo
library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)

# Source your C++ funcitons
sourceCpp("LassoInC.cpp")

# Source your LASSO functions from HW4 (make sure to move the corresponding .R file in the current project folder)
source("LassoFunctions.R")

aeq <- function(x, y, tol = 1e-6) isTRUE(all.equal(x, y, tolerance = tol, check.attributes = FALSE))

set.seed(123)


# Do at least 2 tests for soft-thresholding function below. You are checking output agreements on at least 2 separate inputs
#################################################
cat("=== Tests: soft / soft_c ===\n")
## Test 1: deterministic values incl. edges
vals1   <- c(-3, -1, 0, 0.3, 1.2, 3)
lambda1 <- 1
soft_R1 <- sapply(vals1, function(a) soft(a, lambda1))
soft_C1 <- sapply(vals1, function(a) soft_c(a, lambda1))
stopifnot(aeq(soft_R1, soft_C1))
cat("soft test 1 passed\n")

## Test 2: random values and multiple lambdas
set.seed(1)
vals2 <- rnorm(100)
lams2 <- c(0, 0.1, 0.5, 1.7)
ok_soft <- TRUE
for (lam in lams2) {
  if (!aeq(sapply(vals2, soft, lam), sapply(vals2, soft_c, lam))) ok_soft <- FALSE
}
stopifnot(ok_soft)
cat("soft test 2 passed\n\n")


# Do at least 2 tests for lasso objective function below. You are checking output agreements on at least 2 separate inputs
#################################################
cat("=== Tests: lasso / lasso_c (objective) ===\n")
## Helper: use the same standardization as HW4
stdXY <- function(X, Y) {
  out <- standardizeXY(X, Y)
  list(Xtilde = out$Xtilde, Ytilde = out$Ytilde)
}

## Test 1: small reproducible data
X1 <- matrix(c(1,2,3, 0,1,0, 1,0,1, 0,0,1, 2,1,0), nrow=5, byrow=TRUE)
y1 <- c(1,0,2,0,1)
std1 <- stdXY(X1, y1)
beta1 <- c(0.1, -0.2, 0.3)
lam1  <- 0.4
obj_R1 <- lasso(std1$Xtilde, std1$Ytilde, beta1, lam1)
obj_C1 <- lasso_c(std1$Xtilde, std1$Ytilde, beta1, lam1)
stopifnot(aeq(obj_R1, obj_C1))
cat("lasso objective test 1 passed\n")

## Test 2: random tall data
set.seed(2)
X2 <- matrix(rnorm(300*20), 300, 20)
y2 <- rnorm(300)
std2 <- stdXY(X2, y2)
beta2 <- rnorm(20)
lam2  <- 0.25
obj_R2 <- lasso(std2$Xtilde, std2$Ytilde, beta2, lam2)
obj_C2 <- lasso_c(std2$Xtilde, std2$Ytilde, beta2, lam2)
stopifnot(aeq(obj_R2, obj_C2))
cat("lasso objective test 2 passed\n\n")


# Do at least 2 tests for fitLASSOstandardized function below. You are checking output agreements on at least 2 separate inputs
#################################################
cat("=== Tests: fitLASSOstandardized (single lambda) ===\n")
## Test 1: random data, zero start
set.seed(3)
n <- 200; p <- 50
X3 <- matrix(rnorm(n*p), n, p)
beta_true <- c(rep(2,5), rep(0, p-5))
y3 <- drop(X3 %*% beta_true + rnorm(n, sd = 0.5))
std3 <- stdXY(X3, y3)
lam3 <- 0.3
b0   <- rep(0, p)

fitR3 <- fitLASSOstandardized(std3$Xtilde, std3$Ytilde, lam3, beta_start = b0, eps = 1e-8)
if (is.list(fitR3)) fitR3 <- fitR3$beta
fitC3 <- fitLASSOstandardized_c(std3$Xtilde, std3$Ytilde, lam3, beta_start = b0, eps = 1e-8)

stopifnot(max(abs(as.numeric(fitR3) - as.numeric(fitC3))) < 1e-6)
cat("fit single-lambda test 1 passed\n")


## Test 2: random data, nonzero warm start
set.seed(33)
X4 <- matrix(rnorm(150*30), 150, 30)
y4 <- rnorm(150)
std4 <- stdXY(X4, y4)
lam4 <- 0.15
bstart4 <- rnorm(30, sd = 0.1)

fitR4 <- fitLASSOstandardized(std4$Xtilde, std4$Ytilde, lam4, beta_start = bstart4, eps = 1e-8)
if (is.list(fitR4)) fitR4 <- fitR4$beta
fitC4 <- fitLASSOstandardized_c(std4$Xtilde, std4$Ytilde, lam4, beta_start = bstart4, eps = 1e-8)

stopifnot(max(abs(as.numeric(fitR4) - as.numeric(fitC4))) < 1e-6)
cat("fit single-lambda test 2 passed\n")

# Do microbenchmark on fitLASSOstandardized vs fitLASSOstandardized_c
######################################################################

# Do at least 2 tests for fitLASSOstandardized_seq function below. You are checking output agreements on at least 2 separate inputs
#################################################

# Do microbenchmark on fitLASSOstandardized_seq vs fitLASSOstandardized_seq_c
######################################################################

# Tests on riboflavin data
##########################
require(hdi) # this should install hdi package if you don't have it already; otherwise library(hdi)
data(riboflavin) # this puts list with name riboflavin into the R environment, y - outcome, x - gene erpression

# Make sure riboflavin$x is treated as matrix later in the code for faster computations
class(riboflavin$x) <- class(riboflavin$x)[-match("AsIs", class(riboflavin$x))]

# Standardize the data
out <- standardizeXY(riboflavin$x, riboflavin$y)

# This is just to create lambda_seq, can be done faster, but this is simpler
outl <- fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, n_lambda = 30)

# The code below should assess your speed improvement on riboflavin data
microbenchmark(
  fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, outl$lambda_seq),
  fitLASSOstandardized_seq_c(out$Xtilde, out$Ytilde, outl$lambda_seq),
  times = 10
)
