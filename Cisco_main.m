clear all
close all
clc


M1_All = csvread('UR5_1_all.csv',1,0);
M2_All = csvread('UR5_2_all.csv',1,0);
M3_All = csvread('UR5_3_all.csv',1,0);
M4_All = csvread('UR5_4_all.csv',1,0);

start=2; endpoint=size(M1_All,2)/3;
M1_All=M1_All(:,start:endpoint);
M2_All=M2_All(:,start:endpoint);
M3_All=M3_All(:,start:endpoint);
M4_All=M4_All(:,start:endpoint);

figure, plot(M1_All','b.'), hold on,plot(M2_All','r.'), legend('1 vs 2'), ylim([-6,15])
figure, plot(M1_All','b.'), hold on,plot(M3_All','r.'), legend('1 vs 3'), ylim([-6,15])
figure, plot(M1_All','b.'), hold on,plot(M4_All','r.'), legend('1 vs 4'), ylim([-6,15])
figure, plot(M2_All','b.'), hold on,plot(M3_All','r.'), legend('2 vs 3'), ylim([-6,15])
figure, plot(M2_All','b.'), hold on,plot(M4_All','r.'), legend('2 vs 4'), ylim([-6,15])
figure, plot(M3_All','b.'), hold on,plot(M4_All','r.'), legend('3 vs 4'), ylim([-6,15])

mu1=mean(M1_All);
mu2=mean(M2_All);
mu3=mean(M3_All);
mu4=mean(M4_All);

mu234=mean([mu2;mu3;mu4]);
figure, hold on
plot(mu1,'b')
plot(mu234,'r')

% plot(mu3,'g')
% plot(mu4,'k')

%% PCA
mu=mean(M1_All);
M1_All_Centered = M1_All;%-repmat(mu,size(M1_All,1),1);

[U,S,V]=svd(M1_All_Centered);
M1_All_PC = M1_All_Centered*V;

MOther_All=[M2_All;M3_All;M4_All];
MOther_All_Centered=MOther_All;%-repmat(mu,size(MOther_All,1),1);
MOther_All_PC = MOther_All_Centered*V;

figure,hold on, grid on
plot(ones(6,1),M1_All_PC(:,1),'r*');
plot(ones(18,1),MOther_All_PC(:,1),'bo');

% figure, plot(V(:,1)*50,'g')

figure,hold on, grid on
plot(M1_All_PC(:,1),M1_All_PC(:,2),'ro');
plot(MOther_All_PC(:,1),MOther_All_PC(:,2),'bo');

figure,hold on, grid on
plot3(M1_All_PC(:,1),M1_All_PC(:,2),M1_All_PC(:,3),'ro');
plot3(MOther_All_PC(:,1),MOther_All_PC(:,2),MOther_All_PC(:,3),'bo');

