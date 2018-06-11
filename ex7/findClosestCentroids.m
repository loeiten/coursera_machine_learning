function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Loop over the rows
for i=1:rows(X)
    x = X(i, :);
    min_err = 2^30;
    % Will produce error if not found
    best_k = K + 1;
    for k=1:K
        diff = x - centroids(k, :);
        cur_err = diff*diff';
        if cur_err <= min_err
            min_err = cur_err;
            best_k = k;
        end
    end
    % Assign best_k
    idx(i) = best_k;
end

% =============================================================

end

