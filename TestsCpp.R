
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


# Do at least 2 tests for fitLASSOstandardized function below. You are checking output agreements on at least 2 separate inputs
#################################################

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
