classdef Parameter < handle
    
%   Copyright 2019 Nghia Nguyen (nghia.nguyen@sydney.edu.au)
%   https://github.com/VBayesLab
%   Version: 1.0
%   LAST UPDATE: Sep, 2019

    %PARAMETER Class for model parameters    
    properties
        Name                 % Parameter name
        Prior                % Parameter prior
        PriorTransform       % Transformation of prior 
        PriorInvTransform    % Inverse transform of the prior
        JacobianOffset       % Offset of Jacobian due to the transformation     
    end
    
    methods
        function obj = Parameter(varargin)
            
            obj.Name              = 'Var';
            obj.Prior             = Distribution('Name','Normal','Parameter',[0,0.1]);
            obj.PriorTransform    = @(x)x;
            obj.PriorInvTransform = @(x)x;
            obj.JacobianOffset    = @(x)0;
        
            % User-defined setting
            if nargin > 0
               paramNames = {'Name'              'Prior' ...
                             'PriorTransform'    'PriorInvTransform' ...
                             'JacobianOffset'};
                      
               paramDflts = {obj.Name             obj.Prior ...
                             obj.PriorTransform   obj.PriorInvTransform ...
                             obj.JacobianOffset};

               [obj.Name,...
                obj.Prior,...
                obj.PriorTransform,...
                obj.PriorInvTransform,...
                obj.JacobianOffset] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});
            end
            
        end
    end 
end


