function output = qualityControl(fileName)
% qualityControl - Constructs and runs a bioinformatics quality control pipeline.
%
% BRAD's Example to call this function:
%    Execute: eng.qualityControl("SRR005164_1_50.fastq");
%
% This function creates a bioinformatics quality control pipeline using the
% MATLAB bioinformatics toolbox. It reads a FASTQ file, applies sequence 
% filtering, and generates a quality control plot.
%
% The pipeline consists of the following steps:
% 1. FileChooser block to select the input FASTQ file.
% 2. SeqFilter block to filter the sequences based on quality thresholds.
% 3. UserFunction block to generate a quality control plot of the filtered sequences.
%
% The blocks are connected in sequence and executed in the specified order.
%
% Reference: https://www.mathworks.com/help/bioinfo/ref/bioinfo.pipeline.pipeline.run.html
%
% Auth: Joshua Pickard
%       jpic@umich.edu
% Date: June 14, 2024
    disp('Running QC Pipeline');
    disp('FileName=');
    disp(fileName);
    % Import necessary classes from the bioinformatics toolbox
    import bioinfo.pipeline.Pipeline
    import bioinfo.pipeline.block.*

    % Create an instance of the Pipeline class
    qcpipeline = Pipeline;
    disp('    1. Building the pipeline');
    % Create a FileChooser block to select the input FASTQ file
    fastqfile = bioinfo.pipeline.blocks.FileChooser(which(fileName));

    % Create a SeqFilter block to filter sequences based on quality thresholds
    sequencefilter = bioinfo.pipeline.blocks.SeqFilter;
    sequencefilter.Options.Threshold = [10 20];  % Set quality thresholds for filtering

    % Add the FileChooser and SeqFilter blocks to the pipeline
    addBlock(qcpipeline, [fastqfile, sequencefilter]);

    % Connect the FileChooser block to the SeqFilter block
    connect(qcpipeline, fastqfile, sequencefilter, ["Files", "FASTQFiles"]);

    % Create a UserFunction block to generate a quality control plot
    qcplot = bioinfo.pipeline.blocks.UserFunction("seqqcplot", ...
        RequiredArguments="inputFile", OutputArguments="figureHandle");

    % Add the UserFunction block to the pipeline
    addBlock(qcpipeline, qcplot);

    % Connect the SeqFilter block to the UserFunction block
    connect(qcpipeline, sequencefilter, qcplot, ["FilteredFASTQFiles", "inputFile"]);

    % Add block to save the output to a file
    saveplot = bioinfo.pipeline.blocks.UserFunction("saveplot",RequiredArguments="figureHandle");    
    addBlock(qcpipeline,saveplot);
    connect(qcpipeline,qcplot,saveplot,["figureHandle","figureHandle"]);
    disp('    2. Pipeline Built!');

    % Run the pipeline
    disp('    3. Running Pipeline');
    run(qcpipeline);
    disp('Finished Running QC Pipeline');
    output = 0;
end
