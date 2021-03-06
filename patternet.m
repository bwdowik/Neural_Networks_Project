%% Q2: Data classifier
clear all
clc

N = 20; % number of data points of each class 
offset = 5; % rough measure of distance between classes 
x = [randn(2,N) randn(2,N)+offset]; % inputs 
t(1,:) = [zeros(1,N) ones(1,N)]; % outputs
t(2,:) = [ones(1,N) zeros(1,N)]; % outputs

net = patternnet(2)
net = train(net,x,t);
view(net)
y=net(x);
perf=perform(net,t,y);
classes = vec2ind(y);

% b=net2.b{1} %bias
% W=net2.IW{1} %input weights
% 
% xval = linspace(-5,10);
% yval = (-W(1)*xval-b)/W(2) %using equation Wp+b=0
% 
 figure(1)
 scatter(x(1,1:N),x(2,1:N),'b*') %plot 1st half training data
 hold on
 scatter(x(1,N+1:2*N),x(2,N+1:2*N),'r*') %plot 2nd half training data
% plot(xval,yval,'m') %plot linear boundary
 hold off
 title('Training')
% 
% %% Testing the network
% testSample = [randn(2,N) randn(2,N)+offset]; % inputs
% result = net2(testSample) %running test data through network
% 
% figure(2)
% scatter(testSample(1,1:N),testSample(2,1:N),'b*') %plot 1st half test data
% hold on
% scatter(testSample(1,N+1:2*N),testSample(2,N+1:2*N),'r*') %plot 2nd half test data
% plot(xval,yval,'m') %plot linear boundary
% hold off
% title('Testing')
% 
% 

