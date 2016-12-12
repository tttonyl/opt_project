sigmoid <- function(z) {
  g <- matrix(0,dim(as.matrix(z)))
  g <- 1 / (1 + exp(-1 * z))
  g
  
}
