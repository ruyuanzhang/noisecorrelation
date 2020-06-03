%% Simulate a pool of orientation-selective neurons and try to classify two stimuli
% This script corresponds to Figure 7 A,B,C. Running this script might
% takes sevreal minutes.


clear all;close all;clc;

stim1 = 88;  % deg, set up two stimuli
stim2 = 92;  
nNeurons_list = [10, 20, 50, 100, 200, 400]; % number of neurons in the pool
nTrials_list = nNeurons_list * 100; % number of trials to stimulate
Cneuron = [0, 0.1, 0.3, 0.5, 0.8, 0.99]; % Noise correlation coefficient between neurons
nSimulations = 100;
wantsave = 1; % whether to save all simulation data

% params for the tuning curve
alpha = 1;
beta = 19;
gamma = 2;
orien = 1:180;  % deg, possible orientation stimuli


%% do it
if isempty(gcp) && nSimulations > 1
    pobj=parpool(16);
else
    pobj = gcp;
end

[pCorrect_cTCNC, pCorrect_aTCNC, pCorrect_SFNC]= deal(cell(1,nSimulations));
for iSimu = 1:nSimulations % loop simulations
    iSimu
    
    [pCorrectaTCNCtmp, pCorrectcTCNCtmp, pCorrectSFNCtmp]= deal(zeros(length(Cneuron),length(nNeurons_list)));
    for iNeuron = 1:length(nNeurons_list)
        
        nNeurons = nNeurons_list(iNeuron);
        nTrials = nTrials_list(iNeuron);
        
        phi = 0:180/nNeurons:180;  % deg, prefered orientation
        phi = phi(2:end);
        
        % von mises tuning curve
        [orienxx, phiyy] = meshgrid(orien, phi);
        meanNeuronResp = alpha + beta * exp(gamma*(cos((orienxx - phiyy)*pi/90) - 1));
        
        % mean response
        populationMean1 = meanNeuronResp(:,stim1);
        populationMean2 = meanNeuronResp(:,stim2);
        
        % compute covariance
        % variance matrix
        % note that variance are stimulus dependent
        varMat1 = sqrt(populationMean1) * sqrt(populationMean1');
        varMat2 = sqrt(populationMean2) * sqrt(populationMean2');
        
        for iCneuron = 1:length(Cneuron)            
            % ==== angular-based TCNC ====
            L = 1; % control the width of
            [phix, phiy] = meshgrid(phi, phi);
            orienDiff = abs(circulardiff(phix, phiy, 180));
            R_aTCneuron = Cneuron(iCneuron)*exp(-orienDiff*pi/180/L);            
            R_aTCneuron(logical(eye(size(R_aTCneuron,1)))) = 1;
            Q_aTCneuron1 = varMat1.* R_aTCneuron;
            Q_aTCneuron2 = varMat2.* R_aTCneuron;
            % now generate data and do the classification
            data1 = posrect(mvnrnd(populationMean1, Q_aTCneuron1, nTrials));
            data2 = posrect(mvnrnd(populationMean2, Q_aTCneuron2, nTrials));
            % do the classification
            training = [data1(1:nTrials/2,:); data2(1:nTrials/2,:)];
            sample = [data1(nTrials/2+1:end,:); data2(nTrials/2+1:end,:)];
            group = [ones(nTrials/2,1);2*ones(nTrials/2,1)];
            
            [class,~,~,~,classObj] = classify(sample,training,group,'linear');
            good = 0;
            good = good + sum(class(1:nTrials/2)==1);
            good = good + sum(class(nTrials/2+1:end)==2);
            pCorrectaTCNCtmp(iCneuron,iNeuron) = good/nTrials * 100;
            
            % ==== curve-based TCNC====
            R_SC = corr(meanNeuronResp');   % signal correlation matrix
            R_cTCneuron = Cneuron(iCneuron) * R_SC;
            R_cTCneuron(logical(eye(size(R_cTCneuron, 1)))) = 1; % set diagnal to 1            
            Q_cTCneuron1 = varMat1.*R_cTCneuron;
            Q_cTCneuron2 = varMat2.*R_cTCneuron;
            % now generate data and do the classification
            data1 = posrect(mvnrnd(populationMean1, Q_cTCneuron1, nTrials));
            data2 = posrect(mvnrnd(populationMean2, Q_cTCneuron2, nTrials));
            training = [data1(1:nTrials/2,:); data2(1:nTrials/2,:)];
            sample = [data1(nTrials/2+1:end,:); data2(nTrials/2+1:end,:)];
            group = [ones(nTrials/2,1);2*ones(nTrials/2,1)];            
            %Mdl = fitclinear(training,group,'FitBias',false,'Learner','logistic');        % fit
            %class = predict(Mdl,sample);% test            
            [class,~,~,~] = classify(sample,training,group,'linear');            
            good = 0;
            good = good + sum(class(1:nTrials/2)==1);
            good = good + sum(class(nTrials/2+1:end)==2);
            pCorrectcTCNCtmp(iCneuron,iNeuron) = good/nTrials * 100;
            
            % ==== shuffle NC (SFNC)====
            randOrder = Shuffle(1:nNeurons);
            [xx,yy] = meshgrid(randOrder);
            ind = sub2ind(size(R_SC), xx, yy);
            R_SFNC = R_SC(ind);
            R_SFNC = Cneuron(iCneuron) * R_SFNC;
            R_SFNC(logical(eye(size(R_SFNC,1)))) = 1; % set diagnal to 1
            
            Q_SFNC1 = varMat1.*R_SFNC;
            Q_SFNC2 = varMat2.*R_SFNC;
            % now generate data and do the classification
            data1 = posrect(mvnrnd(populationMean1, Q_SFNC1, nTrials));
            data2 = posrect(mvnrnd(populationMean2, Q_SFNC2, nTrials));
            training = [data1(1:nTrials/2,:); data2(1:nTrials/2,:)];
            sample = [data1(nTrials/2+1:end,:); data2(nTrials/2+1:end,:)];
            group = [ones(nTrials/2,1);2*ones(nTrials/2,1)];

            [class,~,~,~] = classify(sample,training,group,'linear');
            good = 0;
            good = good + sum(class(1:nTrials/2)==1);
            good = good + sum(class(nTrials/2+1:end)==2);
            pCorrectSFNCtmp(iCneuron,iNeuron) = good/nTrials * 100;
        end
    end
    pCorrect_aTCNC{iSimu} = pCorrectcTCNCtmp;
    pCorrect_cTCNC{iSimu} = pCorrectcTCNCtmp;
    pCorrect_SFNC{iSimu} = pCorrectSFNCtmp;
end
clear training sample data1 data2;
if ~isempty(gcp)
    delete(gcp);
end
%% preprocess the data
pCorrect_aTCNC = cat(3,pCorrect_aTCNC{:});
pCorrect_cTCNC = cat(3,pCorrect_cTCNC{:});
pCorrect_SFNC = cat(3, pCorrect_SFNC{:});

mn_aTCneuron = mean(pCorrect_aTCNC, 3);
mn_cTCneuron = mean(pCorrect_cTCNC, 3);
mn_SFNC = mean(pCorrect_SFNC, 3);

%se_aTCneuron = cat(3, zeros(mn_aTCneuron), (prctile(pCorrect_aTCNC,97.5, 3)-prctile(pCorrect_aTCNC,2.5,3))/2);
%se_cTCneuron = cat(3, zeros(mn_cTCneuron), (prctile(pCorrect_cTCNC,97.5, 3)-prctile(pCorrect_cTCNC,2.5,3))/2);
%se_SFNC = cat(3, zeros(mn_SFNC), (prctile(pCorrect_SFNC,97.5, 3)-prctile(pCorrect_SFNC,2.5,3))/2);
%% plot result

% create the labels for figure legends
legend_label_coeff = cell(1,length(Cneuron));
for i=1:length(Cneuron); legend_label_coeff{i}=sprintf('C0=%.2f',Cneuron(i));end
legend_label_nNeuron = cell(1,length(nNeurons_list));
for i=1:length(nNeurons_list); legend_label_nNeuron{i}=sprintf('nNeuron=%d',nNeurons_list(i));end

close all;
h1 = cpsfigure(1,3);
set(h1,'Position',[0 0 1200 300]);

ax(1)=subplot(1,3,1);
%[lh,eh] = myplot(nNeurons_list,mn_aTCneuron, se_aTCneuron,'-');
%[lh, ~] = myplot(nNeurons_list,mn_aTCneuron, [],'-');
%c = cool(length(lh));
%for i=1:length(lh);set(lh(i),'Color',c(i,:));set(eh{i},'Color',c(i,:));end
[lh, ~] = myplot(nNeurons_list,mn_aTCneuron, [],'-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end

xlabel('Number of neurons'); ylabel('Classification accuracy (%)');
set(gca, 'XScale','log');
title('aTCneuron');
legend(legend_label_coeff);

ax(2)=subplot(1,3,2);
%[lh,eh]=myplot(nNeurons_list,mn_cTCneuron,se_cTCneuron,'-');
%c = cool(length(lh));
%for i=1:length(lh);set(lh(i),'Color',c(i,:));set(eh{i},'Color',c(i,:));end

[lh,eh]=myplot(nNeurons_list,mn_cTCneuron,[],'-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of neurons'); ylabel('Classification accuracy (%)');
set(gca, 'XScale','log');
title('cTCneuron');

ax(3)=subplot(1,3,3);
%[lh,eh]=myplot(nNeurons_list, mn_SFNC, se_SFNC,'-');
%c = cool(length(lh));
%for i=1:length(lh);set(lh(i),'Color',c(i,:));set(eh{i},'Color',c(i,:));end
[lh, ~]=myplot(nNeurons_list, mn_SFNC, [], '-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of neurons'); ylabel('Classification accuracy (%)');
set(gca, 'XScale','log');
title('SFNC');
%
h2 = cpsfigure(1,3);
set(h2,'Position',[0 0 1200 300]);

ax(1)=subplot(1,3,1);
%[lh, eh]=myplot(Cneuron,mn_aTCneuron',permute(se_aTCneuron,[2,1,3]),'-');
%c = parula(length(lh));
%for i=1:length(lh);set(lh(i),'Color',c(i,:));set(eh{i},'Color',c(i,:));end
[lh, ~]=myplot(Cneuron,mn_aTCneuron',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cneuron'); ylabel('Classification accuracy (%)');
title('aTCneuron');
legend(legend_label_nNeuron);

ax(2)=subplot(1,3,2);
%[lh, eh]=myplot(Cneuron,mn_cTCneuron',permute(se_cTCneuron,[2,1,3]),'-');
%c = parula(length(lh));
%for i=1:length(lh);set(lh(i),'Color',c(i,:));set(eh{i},'Color',c(i,:));end
[lh, ~]=myplot(Cneuron, mn_cTCneuron',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cneuron'); ylabel('Classification accuracy (%)');
title('cTCneuron');

ax(3)=subplot(1,3,3);
%[lh, eh]=myplot(Cneuron,mn_SFNC',permute(se_SFNC,[2,1,3]),'-');
%c = parula(length(lh));
%for i=1:length(lh);set(lh(i),'Color',c(i,:));set(eh{i},'Color',c(i,:));end

[lh, ~]=myplot(Cneuron,mn_SFNC', [], '-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cneuron'); ylabel('Classification accuracy (%)');
title('SFNC');

%% save the figure
if wantsave
    saveas(h1,'neuronsimu_classify1.fig');
    saveas(h2,'neuronsimu_classify2.fig');    
    print(h1,'-dpdf','-painters','-r300', 'neuronsimu_classify1.pdf');
    print(h2,'-dpdf','-painters','-r300', 'neuronsimu_classify2.pdf');
    close all; save('neuronsimu_classify.mat');
end








