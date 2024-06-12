function ret = matlabPipeline(pipelineName, scPath)
% matlabPipeline - A sample function to demonstrate argument handling
%
% Syntax: matlabPipeline(pipelineName, scPath)
%
% Inputs:
%    pipelineName - name of the pipeline we are running
%    scPath - path to single cell data file
%
% Outputs:
%    None
%
% Example: 
%    matlabPipeline(10, 'example', [1, 2, 3])
%    This example will print the inputs: 10, 'example', and [1, 2, 3]
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% Author: Joshua Pickard
% Email: jpic@umich.edu

%------------- BEGIN CODE --------------

    disp('Argument 1:');
    disp(pipelineName);
    
    disp('Argument 2:');
    disp(scPath);
    
    ret = "WORKING!!"

%------------- END CODE --------------
end
