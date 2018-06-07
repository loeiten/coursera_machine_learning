function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta = X*theta;
% NOTE: Not regularize the bias term
% https://stats.stackexchange.com/questions/86991/reason-for-not-shrinking-the-bias-intercept-term-in-regression?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
J = (1/(2*m))*(h_theta - y)'*(h_theta - y) + (lambda/(2*m))*theta(2:end)'*theta(2:end);

grad = (1/m)*(h_theta - y)'*X;
% Add regularization
grad(:,2:end) += ((lambda/m).*theta(2:end))';
% =========================================================================

grad = grad(:);

end

