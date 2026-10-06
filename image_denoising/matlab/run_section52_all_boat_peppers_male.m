function results = run_section52_all_boat_peppers_male()
%RUN_SECTION52_ALL_BOAT_CLOCK_PEPPERS Run the three configured image cases.
% Each case delegates to its standalone entry point, so lambdaY,
% priorDelta, time limits, solver settings, and time-varying SADMM scales
% remain exactly synchronized with the corresponding individual script.

restoredefaultpath;
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
close all;
clc;

fprintf('Running Section 5.2 cases: boat, peppers,male\n');
fprintf('Each case keeps its own model and solver parameters.\n');

caseRunners = {@run_section52_all_boat_50s, ...
    @run_section52_all_peppers_200s...
    @run_section52_all_male_500s,};
results = struct();
for i = 1:numel(caseRunners)
    caseRunner = caseRunners{i};
    caseResults = caseRunner(false);
    caseNames = fieldnames(caseResults);
    for j = 1:numel(caseNames)
        results.(caseNames{j}) = caseResults.(caseNames{j});
    end
end
end
