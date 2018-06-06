function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% One-hot encode y
y_mat = zeros(size(y, 1), num_labels);
for i=1:m
    % Note: Label 0 is mapped to index 10
    y_mat(i, y(i)) = 1;
end

% Feedforward
z1 = [ones(m, 1) X]; % Adding the bias node
a1 = z1;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1)  a2]; % Adding the bias node
z3 = a2*Theta2';
% NOTE: a3 is our h_theta
a3 = sigmoid(z3);

% NOTE: a3 will have num_labels columns, so will y
%       Assuming a3 and y are vectors (only one class) where the dimension is
%       the number of samples, a dot product would result in the sum of all
%       samples (the sum over m)
%       As a3 and y are matrices, and we are only interested in multiply y and
%       a3 label by label (i.e. the diagonal in the dot product), we will loop
%       over the labels and aggregate a sum (the sum over k)
% NOTE: Recall that the activation function is a sigmoid, which maps y within
%       the range 0 to 1, therefore we have contructed a cost function which
%       has a cost approaching infinity if 1 is predicted close to 0 (the
%       sigmoid prevents zero prediction), and a 0 cost for a correct
%       classification
%       The two terms refers to the two cases if y is 0, or if y is 1
for k=1:num_labels
    J += (1/m) * (-y_mat(:,k)' * log(a3)(:,k) - ...
         (1-y_mat(:,k)') * log(1 -a3(:,k)));
end

% Adding regularization
% NOTE: The regularization is the sum of all the weights squared, excluding the
%       weight of the biases (which is stored in the last column)
regularization = 0;
regularization += sum(sum(Theta1(:, 2:end).*Theta1(:, 2:end)));
regularization += sum(sum(Theta2(:, 2:end).*Theta2(:, 2:end)));
J += (lambda/(2*m))*regularization;

% NOTE: NOTATION
%       Andrew Ng
%       Remember that the weights are residing on the edges of the graph
%       As the network can be seen as a directed graph, we will call the "from"
%       node the previous node, and the "to" node (where the edge is pointing
%       towards) for the next node
%       Theta^{previous layer}_{{next node unit},{previous node unit}}
%
%       Michael Nielsen
%       w^{next layer}_{{next node unit},{previous node unit}}
%
%       We will use Einstein notation (note that ^l is not contravariant index,
%       but layer index)

% NOTE: FORWARD PROP
%       z^l_j = w^l_{j,k} a^{l-1}_k + b^l_j
%       a^l_j = activation(z^l_j)

% NOTE: BACKWARD PROP
%       delta^l_j def= \partial J/\partial z^l_j
%       Is the error from node j in layer l
%       Practical formula:
%       delta^l_j = \partial J/\partial z^l_j
%                 % note different lower index
%                 = \partial J/\partial z^{l+1}_k \partial z^{l+1}_k/\partial z^l_j
%                 = delta^{l+1}_j \partial z^{l+1}_k/\partial z^l_j
%                 = delta^{l+1}_j \partial (w^{l+1}_{k,m} a^l_m + b^{l+1}_k)/\partial z^l_j
%                 = delta^{l+1}_j \partial (w^{l+1}_{k,m} activation(z^l_m) + b^{l+1}_k)/\partial z^l_j
%                 % note new index due to all terms which are zero after differentiation
%                 = delta^{l+1}_j w^{l+1}_{k,j} \partial activation(z^l_j) /\partial z^l_j
%       In the end, we would like to calculate
%       \partial J/\partial w^l_{j,k}
%       If write out formula for cost, will see that
%       J = J ( a^L_j (z^L_j (a^{L-1}_j (z^{L-1}_j ( ... a^{l+1}_j (z^{l+1}_j( a^l_j (z^l_j(w^l_{j,k}) ) ) ) ) ) ) ) )
%       By using chain rule, we see that
%       \partial J/\partial a^L_j ... \partial a^{l}_j/\partial z^l_j = \partial J/\partial z^{l}_j = delta delta^l_j
%       Furthermore, the last factor
%       \partial z^l_j/\partial w^l_{j,k} = a^{l-1}_j
%       we have
%       \partial J/\partial w^l_{j,k} = delta^l_j a^{l-1}_j
%
%       See
%       http://neuralnetworksanddeeplearning.com/chap2.html
%       https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

% NOTE: With the equation above, it is not clear where the delta^3 formula
%       comes from, as it appears that it cannot be derived from delta^l_j rule
%       with the loss function we have implemented (however, if we use the
%       squared error, it pops out from the equations)
%       http://www.wolframalpha.com/input/?i=d%2Fdz+(-y*log(a(z))+-(1-y)*log(a(z)))
%       Other than that, the descrepancy between the delta in Lecture 9, slide
%       7 in Ng and in equation 45 in Nielsen, can be explained by the
%       different indexing of the layers

% Backprop
% Step 2: For one training example (here stored as row), subtract each
%         prediction from the true label
delta3 = a3 - y_mat;

% Step 3:
delta2 = (Theta2'*delta3');
% delta2 now contains the bias added in a2
% As we are element mutliplying with z2 (which misses this extra bias column), we
% remove this row from delta2
delta2 = delta2(2:end,:)'.*sigmoidGradient(z2);

% Step 4 and 5:
% NOTE: We do not need the Delta step as we are doing vectorized calculations
Theta1_grad = (1/m)*delta2'*a1;
Theta2_grad = (1/m)*delta3'*a2;

% Adding regularization
% We skip the first column as this is the bias column
Theta1_grad += (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2_grad += (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:, 2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

