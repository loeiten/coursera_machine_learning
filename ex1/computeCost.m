function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% J = (1/(2m))*sum(h_theta(x^(i)) - y^(i))^2
% In our special case, we have
% h_theta = theta_0*x_0 + theta_1*x_1, where x_0 = 1, so we can express this as
% h_theta = X*theta, where X is a mx2 matrix, and theta is a 2x1 matrix, where m
% is the number os samples

% This means that
% sum(h_theta(x^(i)) - y^(i))^2 = (h_theta_vec - y_vec)^T*(h_theta_vec - y_vec)
% so

delta = X*theta - y;
J = (1/(2*m))*(delta)'*(delta);

% =========================================================================

end

