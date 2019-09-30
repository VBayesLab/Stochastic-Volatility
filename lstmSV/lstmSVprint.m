function lstmSVprint(theta,iter)
%LSTMSVPRINT Custom printed message during the sampling phase

disp(['iteration: ',num2str(iter), '|beta0: ',num2str(theta.beta0),'|beta1: ',num2str(theta.beta1), ...
     '|phi: ',num2str(theta.phi),'|sigma2: ',num2str(theta.sigma2)]);    

end

