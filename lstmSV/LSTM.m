classdef LSTM < handle
    %LSTM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        h
        eta
        memoryCell   
        DimState
        Params
    end
    
    methods
        function obj = LSTM(NumPaths,numObs,theta)
            %LSTM Construct an instance of the LSTM class
            obj.h          = zeros(NumPaths,numObs);
            obj.eta        = zeros(NumPaths,numObs);
            obj.eta(:,1)   = theta.beta0 + normrnd(0,sqrt(theta.sigma2),NumPaths,1);
            obj.memoryCell = zeros(NumPaths,numObs);
            obj.DimState   = NumPaths;
        end
        
        function obj = forward(obj,theta,idx)
            %FORWARD Calculate output of a LSTM cell given previous cell
            %state, cell output and eta 
            a_d = theta.v_d.*obj.eta(:,idx) + theta.w_d.*obj.h(:,idx) + theta.b_d;
            a_i = theta.v_i.*obj.eta(:,idx) + theta.w_i.*obj.h(:,idx) + theta.b_i;
            a_o = theta.v_o.*obj.eta(:,idx) + theta.w_o.*obj.h(:,idx) + theta.b_o;
            a_f = theta.v_f.*obj.eta(:,idx) + theta.w_f.*obj.h(:,idx) + theta.b_f;

            z_d = utils.activation(a_d,'Tanh');
            g_i = utils.activation(a_i,'Sigmoid');
            g_o = utils.activation(a_o,'Sigmoid');
            g_f = utils.activation(a_f,'Sigmoid');

            obj.memoryCell(:,idx+1) = g_f.*obj.memoryCell(:,idx) + g_i.*z_d;
            obj.h(:,idx+1) = g_o.*tanh(obj.memoryCell(:,idx+1));
            obj.eta(:,idx+1) = theta.beta0 + theta.beta1*obj.h(:,idx+1) + normrnd(0,sqrt(theta.sigma2),obj.DimState,1);
        end
        
        % Shuffle lstm cell parameters with given indexes
        function obj = resampling(obj,idx,resampleIdx)
            obj.eta(:,idx)        = obj.eta(resampleIdx,idx);
            obj.h(:,idx)          = obj.h(resampleIdx,idx);
            obj.memoryCell(:,idx) = obj.memoryCell(resampleIdx,idx);
        end
    end
end

