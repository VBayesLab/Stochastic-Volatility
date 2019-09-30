classdef lstmSV < handle & matlab.mixin.CustomDisplay & SV
    %LSTMSV Class to store attributes and methods for lstmSV model
    
    % Class properties
    properties
        LstmCell           % Object of LSTM class
    end
    
    % Method for object display
    methods (Access = protected)
       % Display object properties
       function propgrp = getPropertyGroups(~)
          proplist = {'Distribution','NumParams'};
          propgrp = matlab.mixin.util.PropertyGroup(proplist);
       end
       
       % Display footer
       function header = getHeader(obj)
          if ~isscalar(obj)
             header = getHeader@matlab.mixin.CustomDisplay(obj);
          else
             newHeader1 = '    LSTM-SV Stochastic Volatility Model:';
             newHeader2 = '    ------------------------------------';
             newHeader = {newHeader1
                          newHeader2};
             header = sprintf('%s\n',newHeader{:});
          end
      end
    end
    
    % Store methods for lstmSV class
    methods     
        %% Constructor
        function obj = lstmSV(varargin)
            obj.ModelName      = 'LSTM-SV';
            obj.Distribution   = 'Gaussian';
            obj.Params         = [];     % Clear Params created from superclass's constructor
            obj.Params.beta0   = Parameter('Name','beta0',...
                                           'Prior', Distribution('Name','Normal','Parameter',[0,0.01]));
            obj.Params.beta1   = Parameter('Name','beta1' ,...
                                           'Prior', Distribution('Name','IG','Parameter',[2.5,0.25]));
            obj.Params.phi     = Parameter('Name','phi'   ,...
                                           'Prior', Distribution('Name','Beta','Parameter',[20,1.5]),...
                                           'PriorTransform'   ,@(x)(x+1)/2,...
                                           'PriorInvTransform',@(x)2*x-1,...
                                           'JacobianOffset',@(x)log(0.5));
            obj.Params.sigma2  = Parameter('Name','sigma2',...
                                           'Prior', Distribution('Name','IG','Parameter',[2.5,0.25]));
            obj.Params.v_d     = Parameter('Name','v_d');
            obj.Params.w_d     = Parameter('Name','w_d');
            obj.Params.b_d     = Parameter('Name','b_d');
            obj.Params.v_i     = Parameter('Name','v_i');
            obj.Params.w_i     = Parameter('Name','w_i');
            obj.Params.b_i     = Parameter('Name','b_i');
            obj.Params.v_o     = Parameter('Name','v_o');
            obj.Params.w_o     = Parameter('Name','w_o');
            obj.Params.b_o     = Parameter('Name','b_o');
            obj.Params.v_f     = Parameter('Name','v_f');
            obj.Params.w_f     = Parameter('Name','w_f');
            obj.Params.b_f     = Parameter('Name','b_f');
            obj.MeasurementFcn = @lstmSVmeasurementFnc;
            obj.StateTransitionFcn = @lstmSVstateTransitionFnc;
            obj = obj.setPropertiesFnc();
        end
        
        %% Simulate from a lstmSV model
        function [V,Y,Theta] = simulate(obj,numObs,varargin)
            [V,Y,Theta] = lstmSVsimulate(obj,numObs,varargin);
        end
        
        %% Initialize a parameter arrays from priors
        function theta_init = initialize(obj)
            % By default, initialize parameters using their priors 
            params      = obj.Params;
            num_params  = obj.NumParams;
            params_name = obj.NameParams;

            for i=1:num_params
                theta_init.(params_name{i}) = params.(params_name{i}).Prior.randomGeneratorFnc();
            end
        end
        
        %% Plot useful figures
        function plot(obj,type,varargin)
            lstmSVplot(obj,type,varargin);
        end
        
        %% Make forecast with a lstmSV model
        function forecast_out = forecast(obj,varargin)
            forecast_out = lstmSVforecast(obj,varargin);
        end
        
        %% Print function
        function PrintMsg(obj,theta,i,varargin)
            lstmSVprint(theta,i);
        end
        
    end
    
end

