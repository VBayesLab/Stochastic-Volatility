function GMModel = IS2proposal(theta,distribution)
%IS2PROPOSAL 
if (strcmp(distribution,'GM'))
    options = statset('MaxIter',1000);
    GMModel = fitgmdist(theta,3,'RegularizationValue',0.1,'Options',options);
    % Sample M samples from the fitted mixture model
%     theta_samples = random(GMModel,1);
else
end

