%% Simulate a pool of neurons on a stimulus-estimation task
% This script correponds to Figure 4 D,E,F
% Running this script takes several hours!

clear all;close all;clc;

nNeurons_list = [10, 20, 50, 100, 200, 400];
Cneuron = [0, 0.1, 0.3, 0.5, 0.7, 0.9]; % NC coefficient
nTrials= 1000; 
wantsave = 1;
stim = ceil((rand(1,nTrials)*180));

% params for the tuning curve
alpha = 1;
beta = 19;
gamma = 2;
orien = 1:180;  % deg, possible orientation stimuli

% open multi cores
if isempty(gcp)
    pobj = parpool(20);
end

[estVar_aTCNC, estVar_cTCNC, estVar_SFNC] = deal(zeros(length(Cneuron),length(nNeurons_list)));
for iNeuron = 1:length(nNeurons_list)
    iNeuron
    
    nNeurons = nNeurons_list(iNeuron);
    % generate tuning curves
    phi = 0:180/nNeurons:180;  % deg, prefered orientation
    phi = phi(2:end);
    % von mises tuning curve
    [orienxx, phiyy] = meshgrid(orien, phi);
    meanNeuronResp = alpha + beta * exp(gamma*(cos((orienxx - phiyy)*pi/90) - 1)); % nNeurons x nOrientation responses
    
    for iCneuron = 1:length(Cneuron)
        iCneuron
        
        % create the noise correlation matrix
        % ==== aTCNC ====
        L = 1; % control the 
        [phix, phiy] = meshgrid(phi, phi);
        orienDiff = abs(circulardiff(phix, phiy, 180));
        R_aTCneuron = Cneuron(iCneuron)*exp(-orienDiff*pi/180/L);
        R_aTCneuron(logical(eye(size(R_aTCneuron,1)))) = 1;
        
        % ==== cTCNC ====
        R_SC = corr(meanNeuronResp');   % signal correlation matrix
        R_cTCneuron = Cneuron(iCneuron) * R_SC;
        R_cTCneuron(logical(eye(size(R_cTCneuron, 1)))) = 1; % set diagnal to 1
        
        % ==== SFNC ====
        randOrder = Shuffle(1:nNeurons);
        [xx,yy] = meshgrid(randOrder);
        ind = sub2ind(size(R_SC), xx, yy);
        R_SFNC = R_SC(ind);
        R_SFNC = Cneuron(iCneuron) * R_SFNC;
        R_SFNC(logical(eye(size(R_SFNC,1)))) = 1; % set diagnal to 1
        
        % now do the maximum likelihood estimator
        [logpdf_aTCneuron, logpdf_cTCneuron, logpdf_SFNC]= deal(zeros(nTrials,180));
        
        parfor j = 1:nTrials %loop all trials
            
            % generate data
            Q_aTCneuron = diag(sqrt(meanNeuronResp(:,stim(j))))*R_aTCneuron*diag(sqrt(meanNeuronResp(:,stim(j))));
            data_aTCneuron = posrect(mvnrnd(meanNeuronResp(:,stim(j)), Q_aTCneuron));
            
            Q_cTCneuron = diag(sqrt(meanNeuronResp(:,stim(j))))*R_cTCneuron*diag(sqrt(meanNeuronResp(:,stim(j))));
            data_cTCneuron = posrect(mvnrnd(meanNeuronResp(:,stim(j)), Q_cTCneuron));
            
            Q_SFNC = diag(sqrt(meanNeuronResp(:,stim(j))))*R_SFNC*diag(sqrt(meanNeuronResp(:,stim(j))));
            data_SFNC = posrect(mvnrnd(meanNeuronResp(:,stim(j)), Q_SFNC));
            
            for i=1:180  % do grid search for the loglikelihood of the stimulus given the results
                u = meanNeuronResp(:,i); % the mean of the multivariate responses
                
                covtmp_aTCneuron = diag(sqrt(u)) * R_aTCneuron * diag(sqrt(u));
                logpdf_aTCneuron(j,i) = -1/2*log(det(covtmp_aTCneuron))-(data_aTCneuron'-u)' / covtmp_aTCneuron * (data_aTCneuron'-u);
                
                covtmp_cTCneuron = diag(sqrt(u)) * R_cTCneuron * diag(sqrt(u));
                logpdf_cTCneuron(j,i) = -1/2*log(det(covtmp_cTCneuron))-(data_cTCneuron'-u)' / covtmp_cTCneuron * (data_cTCneuron'-u);
                
                covtmp_SFNC = diag(sqrt(u)) * R_SFNC * diag(sqrt(u));
                logpdf_SFNC(j,i) = -1/2*log(det(covtmp_SFNC))-(data_SFNC'-u)' / covtmp_SFNC * (data_SFNC'-u);
            end
        end
        % take the maximual likelihood, get the estimated stimulus 
        [~,y_pred] = max(logpdf_aTCneuron,[],2);
        estVar_aTCNC(iCneuron, iNeuron) = circularsqerror(y_pred, stim', 180) / nTrials;
        
        [~,y_pred] = max(logpdf_cTCneuron,[],2);
        estVar_cTCNC(iCneuron, iNeuron) = circularsqerror(y_pred, stim', 180) / nTrials;
        
        [~,y_pred] = max(logpdf_SFNC,[],2);
        estVar_SFNC(iCneuron, iNeuron) = circularsqerror(y_pred, stim', 180) / nTrials;
    end
end

if ~isempty(gcp)
    delete(gcp);
end
%% plot result
% make legend label
legend_label = cell(1,length(Cneuron));
for i=1:length(Cneuron); legend_label{i}=sprintf('C0=%.2f',Cneuron(i));end
legend_label_nNeuron = cell(1,length(nNeurons_list));
for i=1:length(nNeurons_list); legend_label_nNeuron{i}=sprintf('nNeuron=%d',nNeurons_list(i));end

close all;
h1=cpsfigure(1,3); % 
set(h1,'Position',[0 0 1200 300]);
ax(1)=subplot(1,3,1);
[lh,~]=myplot(nNeurons_list,1./estVar_aTCNC,[],'-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of neurons'); ylabel('MLE efficiency');
set(gca, 'XScale','log');
title('aTCneuron');
legend(legend_label);

ax(2)=subplot(1,3,2);
[lh,~]=myplot(nNeurons_list,1./estVar_cTCNC,[],'-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of neurons'); ylabel('MLE efficiency');
set(gca, 'XScale','log');
title('cTCneuron');

ax(3)=subplot(1,3,3);
[lh,~]=myplot(nNeurons_list,1./estVar_SFNC,[],'-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of neurons'); ylabel('MLE efficiency');
set(gca, 'XScale','log');
title('SFNC');

h2 = cpsfigure(1,3);
set(h2,'Position',[0 0 1200 300]);
ax(1)=subplot(1,3,1);
[lh,~]=myplot(Cneuron,1./estVar_aTCNC',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('NC coef'); ylabel('MLE efficiency');
set(gca, 'YScale','log');
title('aTCneuron');
legend(legend_label_nNeuron);

ax(2)=subplot(1,3,2);
[lh,~]=myplot(Cneuron,1./estVar_cTCNC',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('NC coef'); ylabel('MLE efficiency');
set(gca,'YScale','log');
title('cTCneuron');

ax(3)=subplot(1,3,3);
[lh,~]=myplot(Cneuron,1./estVar_SFNC',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cneuron'); ylabel('MLE efficiency');
set(gca, 'YScale','log');
title('SFNC');

%% save the data and figure
if wantsave
    saveas(h1,'neuronsimu_estimationMLE1.fig');
    saveas(h2,'neuronsimu_estimationMLE2.fig');    
    print(h1,'-dpdf','-painters','-r300','neuronsimu_estimationMLE1.pdf');
    print(h2,'-dpdf','-painters','-r300','neuronsimu_estimationMLE2.pdf');
    close all; save('neuronsimu_estimationMLE');
end











