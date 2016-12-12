predict <- function(Theta1, Theta2, X) {
  if (is.vector(X))
    X <- t(X)
  m <- dim(X)[1]
  num_labels <- dim(Theta2)[1]
  p <- rep(0,dim(X)[1])
  
  h1 <- sigmoid(cbind(rep(1,m),X) %*% t(Theta1))
  h2 <- sigmoid(cbind(rep(1,m),h1) %*% t(Theta2))
  
  p <- apply(h2,1,which.max)
  p

}
