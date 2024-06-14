%SVD    Singular value decomposition.
%   [U,S,V] = SVD(X) produces a diagonal matrix S, of the same 
%   dimension as X and with nonnegative diagonal elements in
%   decreasing order, and unitary matrices U and V so that
%   X = U*S*V'.
%
%   S = SVD(X) returns a vector containing the singular values.
%
%   [U,S,V] = SVD(X,"econ") produces the "economy size"
%   decomposition. If X is m-by-n then
%     m > n  - only the first n columns of U are computed, S is n-by-n.
%     m == n - equivalent to SVD(X)
%     m < n  - only the first m columns of V are computed, S is m-by-m.
%
%   [U,S,V] = SVD(X,0) is equivalent to SVD(X, "econ") if X has more rows
%   than columns (m > n). Otherwise, it is equivalent to SVD(X).
%    Note: This syntax is not recommended. Use the "econ" option instead.
%
%   [...] = svd(...,sigmaForm) returns singular values in the form
%   specified by sigmaForm using any of the previous input or output
%   argument combinations. sigmaForm can be "vector" to return the singular
%   values in a vector, or "matrix" to return them in a diagonal matrix.
%
%   See also SVDS, GSVD, PAGESVD.

%   Copyright 1984-2022 The MathWorks, Inc.
%   Built-in function.
