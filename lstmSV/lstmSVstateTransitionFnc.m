function NewState = lstmSVstateTransitionFnc(Model,PreviousState,varargin)
%LSTMSVSTATETRANSITION Propose current state given previous state

eta(:,t) = Model.beta0 + Model.beta1*normrnd(0,sqrt(Model.sigma2),N,1);
NewState(:,t) = eta(:,t) + mdl.phi*PreviousState;

end

