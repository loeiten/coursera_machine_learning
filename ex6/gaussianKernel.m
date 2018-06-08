function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

% Recall:
% We classify y=1 if sum theta_i x_i geq 0
% In SVM we transform x_i to a f_i
% In this case we are using a Radial Basis Function kernel
% So, for finding theta:
% 1. We take our training set X, transform it to F
% 2. Invert the problem F*theta = 0
%    i. Inversion can be performed by pseudo-inverting F
%    ii. Inversion can be done by gradient descent (cost function minimizing)
%        Remember that the cost function is a scalar (it usually contain a
%        y*(X*theta) product somewhere)

sim = exp(-((x1-x2)'*(x1-x2))/(2*sigma^2));

% =============================================================

end

