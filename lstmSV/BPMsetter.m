function obj = BPMsetter(obj,varargin)
%BPMSETTER Set up user-defined settings for BPM sampler
%   Copyright 2019 Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%
%   https://github.com/VBayesLab
%
%   Version: 1.0
%   LAST UPDATE: May, 2019

%% Parse additional options
paramNames = {'NumParticles'    'NumMCMC'         'TargetAccept'      'BlockSize'   ...
              'ParamsInit'      'SaveFileName'    'SaveAfter'         'Verbose'     ...
              'NumCovariance'   'Seed'            'ModelToFit'        'log_likelihood'};
paramDflts = {obj.NumParticles  obj.NumMCMC       obj.TargetAccept    obj.BlockSize  ...
              obj.ParamsInit    obj.SaveFileName  obj.SaveAfter       obj.Verbose   ...
              obj.NumCovariance obj.Seed          obj.ModelToFit      obj.LogLikelihood};

 [obj.NumParticles,...
  obj.NumMCMC,...
  obj.TargetAccept,...
  obj.BlockSize,...
  obj.ParamsInit,...
  obj.SaveFileName,...
  obj.SaveAfter,...
  obj.Verbose,...
  obj.NumCovariance,...
  obj.Seed,...
  obj.ModelToFit,...
  obj.LogLikelihood] = internal.stats.parseArgs(paramNames, paramDflts, varargin{1}{2}{:});

if (~isempty(obj.ParamsInit))
    obj.Initialize = 'custom';
end
end

