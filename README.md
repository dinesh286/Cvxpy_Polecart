# Cvxpy_Polecart
The code to generate optimal control input for a pole cart problem.  The details of the problem is described in https://www.youtube.com/watch?v=wlkRYMVUZTs.  The code is deloped based on the code available in http://www.philipzucker.com/cart-pole-trajectory-optimization-using-cvxpy/.   


The problem is solved by sucessive convex programing approach.  Cvxpy is used for solving problem.  Hermit-simpson collocation is used for intergation.  The code gives optimized result better than the original code.  But it takes more number of iterations (~190 iterations) to solve the problem.  Also it takes more time to run the code (~20 min).  

Suggestions are invited to improve the coding style to reduce number of iteration and run time in a standart PC
