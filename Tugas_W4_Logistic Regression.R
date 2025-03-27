#Fungsi Regresi Logistik 
#Metode Newton-Rhapson & IRLS

logistic_regression = function(X,y,method="NR",tol=1e-5,max.iter=1000){
  beta = rep(0,ncol(X))
  p = 1/(1+exp(-X%*%beta))
  
  gradient = function(beta,X,y){
    p = 1/(1+exp(-X%*%beta))
    grad = t(X) %*% (y-p)
    return(grad)
  }
  
  hessian = function(beta,X){
    p = 1/(1+exp(-X%*%beta))
    W = diag(as.vector(p*(1-p)))
    H = -t(X) %*% W %*% X
    return(H)
  }
  
  
  for (i in 1:max.iter) {
    if (method == "NR") {
      p = 1/(1+exp(-X %*% beta))
      grad = gradient(beta,X,y)
      H = hessian(beta,X)
      beta_new = beta - solve(H) %*% grad
    } 
    
    else if (method == "IRLS") {
      p = 1/(1+exp(-X %*% beta))
      W = diag(as.vector(p*(1-p)))
      z = X %*% beta + solve(W) %*% (y-p)
      xtw = t(X) %*% W
      xtwx_inv = solve(t(X) %*% W %*% X)
      beta_new = xtwx_inv %*% (xtw %*% z)
    } 
    
    else {
      stop("Method Not Detected, Try Again.")
    }
    
    if(sqrt(sum((beta_new-beta)^2))<tol){
      cat('Converged in',i,'iterations\n')
      return(list(method=method, beta=beta_new, fit=p))
    }
    
    beta = beta_new
  }
  
  warning("Reached Max Iteration Without Full Convergence")
  return(list(method=method, beta=beta, fit=p))
}

#Contoh Aplikasi
X = matrix(rnorm(100),ncol=2)
y = rbinom(50,1,0.5)

#Penggunaan NR
NR_Result = logistic_regression(X,y,method="NR")
print(NR_Result)

#Penggunaan IRLS
IRLS_Result = logistic_regression(X,y,method="IRLS")
print(IRLS_Result)
