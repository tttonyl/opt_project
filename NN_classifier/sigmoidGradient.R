sigmoidGradient <- function(z) {

  g  <- sigmoid(z) * (1 - sigmoid(z))
  g
}
