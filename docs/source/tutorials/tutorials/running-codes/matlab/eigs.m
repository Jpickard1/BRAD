function [V, D, flag] = eigs(varargin)
% EIGS   Find a few eigenvalues and eigenvectors of a matrix
%  D = EIGS(A) returns a vector of A's 6 largest magnitude eigenvalues.
%  A must be square and should be large and sparse.
%
%  [V,D] = EIGS(A) returns a diagonal matrix D of A's 6 largest magnitude
%  eigenvalues and a matrix V whose columns are the corresponding
%  eigenvectors.
%
%  [V,D,FLAG] = EIGS(A) also returns a convergence flag. If FLAG is 0 then
%  all the eigenvalues converged; otherwise not all converged.
%
%  EIGS(A,B) solves the generalized eigenvalue problem A*V == B*V*D. B must
%  be the same size as A. EIGS(A,[],...) indicates the standard eigenvalue
%  problem A*V == V*D.
%
%  EIGS(A,K) and EIGS(A,B,K) return the K largest magnitude eigenvalues.
%
%  EIGS(A,K,SIGMA) and EIGS(A,B,K,SIGMA) return K eigenvalues. If SIGMA is:
%
%      'largestabs' or 'smallestabs' - largest or smallest magnitude
%    'largestreal' or 'smallestreal' - largest or smallest real part
%                     'bothendsreal' - K/2 values with largest and
%                                      smallest real part, respectively
%                                      (one more from largest if K is odd)
%
%  For nonsymmetric problems, SIGMA can also be:
%    'largestimag' or 'smallestimag' - largest or smallest imaginary part
%                     'bothendsimag' - K/2 values with largest and
%                                     smallest imaginary part, respectively
%                                     (one more from largest if K is odd)
%
%  If SIGMA is a real or complex scalar including 0, EIGS finds the
%  eigenvalues closest to SIGMA.
%
%  EIGS(A,K,SIGMA,NAME,VALUE) and EIGS(A,B,K,SIGMA,NAME,VALUE) configures
%  additional options specified by one or more name-value pair arguments:
%
%  'IsFunctionSymmetric' - Symmetry of matrix applied by function handle Afun
%            'Tolerance' - Convergence tolerance
%        'MaxIterations' - Maximum number of iterations
%    'SubspaceDimension' - Size of subspace
%          'StartVector' - Starting vector
%     'FailureTreatment' - Treatment of non-converged eigenvalues
%              'Display' - Display diagnostic messages
%           'IsCholesky' - B is actually its Cholesky factor
%  'CholeskyPermutation' - Cholesky vector input refers to B(perm,perm)
%  'IsSymmetricDefinite' - B is symmetric positive definite
%
%  EIGS(A,K,SIGMA,OPTIONS) and EIGS(A,B,K,SIGMA,OPTIONS) alternatively
%  configures the additional options using a structure. See the
%  documentation for more information.
%
%  EIGS(AFUN,N) and EIGS(AFUN,N,B) accept the function AFUN instead of the
%  matrix A. AFUN is a function handle and Y = AFUN(X) should return
%     A*X            if SIGMA is unspecified, or a string other than 'SM'
%     A\X            if SIGMA is 0 or 'SM'
%     (A-SIGMA*I)\X  if SIGMA is a nonzero scalar (standard problem)
%     (A-SIGMA*B)\X  if SIGMA is a nonzero scalar (generalized problem)
%  N is the size of A. The matrix A, A-SIGMA*I or A-SIGMA*B represented by
%  AFUN is assumed to be nonsymmetric unless specified otherwise
%  by OPTS.issym.
%
%  EIGS(AFUN,N,...) is equivalent to EIGS(A,...) for all previous syntaxes.
%
%  Example:
%     A = delsq(numgrid('C',15));  d1 = eigs(A,5,'SM');
%
%  Equivalently, if dnRk is the following one-line function:
%     %----------------------------%
%     function y = dnRk(x,R,k)
%     y = (delsq(numgrid(R,k))) \ x;
%     %----------------------------%
%
%     n = size(A,1);  opts.issym = 1;
%     d2 = eigs(@(x)dnRk(x,'C',15),n,5,'SM',opts);
%
%  See also EIG, SVDS, FUNCTION_HANDLE.

%  Copyright 1984-2022 The MathWorks, Inc.
%  References:
%  [1] Stewart, G.W., "A Krylov-Schur Algorithm for Large Eigenproblems."
%  SIAM. J. Matrix Anal. & Appl., 23(3), 601-614, 2001.
%  [2] Lehoucq, R.B. , D.C. Sorensen and C. Yang, ARPACK Users' Guide,
%  Society for Industrial and Applied Mathematics, 1998


% Error check inputs and derive some information from them
[A, n, B, k, Amatrix, eigsSigma, shiftAndInvert, cholB, permB, scaleB, spdB,...
    innerOpts, useEig, originalB] = checkInputs(varargin{:});

if ~useEig || ismember(innerOpts.method, {'LI', 'SI'})
    % Determine whether B is HPD and do a Cholesky decomp if necessary
    [R, permB, spdB] = CHOLfactorB(B, cholB, permB, shiftAndInvert, spdB);
    
    % Final argument checking before call to algorithm
    [innerOpts, useEig, eigsSigma] = extraChecks(innerOpts, B, n, k, spdB, useEig, eigsSigma);
end

if innerOpts.disp
    displayInitialInformation(innerOpts, B, n, k, spdB, useEig, eigsSigma, shiftAndInvert, Amatrix);
end
