function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

best_err = 2^30;

Cs = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmas = [0.01 0.03 0.1 0.3 1 3 10 30];

for cur_C = Cs
    for cur_sigma = sigmas
        fprintf(['Testing C = %f and sigma = %f.\n'], cur_C, cur_sigma);
        model = svmTrain(X, y, cur_C, ...
                         @(x1, x2) gaussianKernel(x1, x2, cur_sigma));
        predictions = svmPredict(model, Xval);
        cur_err = mean(double(predictions ~= yval));
        fprintf(['Best error = %f and cur error = %f.\n'], best_err, cur_err);
        if cur_err <= best_err
            fprintf('Updating the best accuracy.\n');
            best_err = cur_err;
            C = cur_C;
            sigma = cur_sigma
        end
        fprintf('\n\n');
    end
end

% =========================================================================

end

