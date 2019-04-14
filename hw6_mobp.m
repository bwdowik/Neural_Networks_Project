%% HW6_part 3 MOBP

clear all
clc
clf

syms p;
syms G(p);
G(p) = 1 + sin(pi*p/2);
p_vals = linspace(-2,2,80);
alpha = 0.1;
gamma = 0.1;
batch_size=5;
k=1;
target = double(G(p_vals));

%random initial weights, will change each time, might want to not use for testing
W1(:,k) = [rand(1)-0.5; rand(1)-0.5];
b1(:,k) = [rand(1)-0.5; rand(1)-0.5];
W2(k,:) = [rand(1)-0.5 rand(1)-0.5];
b2(k) = [rand(1)-0.5];

% might be better to use these for testing
% W1(:,k) = [-0.27; -0.41];
% b1(:,k) = [-0.48; -0.13];
% W2(k,:) = [0.09 -0.17];
% b2(k) = [0.48];

deltaW1(:,k) = [0; 0];
deltab1(:,k) = [0; 0];
deltaW2(k,:) = [0 0];
deltab2(k) = [0];

%trying to modify for batching
index=1;
rand_index = randperm(length(p_vals));

for k=1:1000
    W2sum=0;
    b2sum=0;
    W1sum=0;
    b1sum=0;
    
    for q=1:batch_size
        %This stuff is so we just train it with randomly ordered values
        p = p_vals(rand_index(index));
        index=index+1;
        if index>=length(p_vals)
           rand_index = randperm(length(p_vals));
           index=1; 
        end

        t = double(G(p));
        a0=p;

        a1=logsig(W1(:,k)*a0+b1(:,k));
        a2=W2(k,:)*a1+b2(k);
        
        e = t-a2;

        s2 = -2*f2()*e;

        F1n1 = zeros(2);
        F1n1(1,1) = f1(a1(1));
        F1n1(2,2) = f1(a1(2));
        s1 = F1n1*W2(k,:)'*s2;
        
        W2sum=W2sum+(s2*a1');
        b2sum=b2sum+s2;
        W1sum=W1sum+s1*a0';
        b1sum=b1sum+s1;
    end
    
    deltaW2(k+1,:)=gamma*deltaW2(k,:) - (1-gamma)*alpha*W2sum/batch_size;
    deltab2(k+1) = gamma*deltab2(k) - (1-gamma)*alpha*b2sum/batch_size;
    deltaW1(:,k+1) = gamma*deltaW1(:,k) - (1-gamma)*alpha*W1sum/batch_size;
    deltab1(:,k+1) = gamma*deltab1(:,k) - (1-gamma)*alpha*b1sum/batch_size;
    
    W2(k+1,:) = W2(k,:)+deltaW2(k+1,:);
    b2(k+1) = b2(k)+deltab2(k+1);
    W1(:,k+1) = W1(:,k)+deltaW1(:,k+1);
    b1(:,k+1)=b1(:,k)+deltab1(:,k+1);
    
    netoutput=myNet(p_vals, W1(:,k+1),W2(k+1,:),b1(:,k+1),b2(k+1));
    error(k) = sum((target-netoutput).^2)/length(target);   
end

%% plots
figure(1)
hold on
finalOutput=myNet(p_vals, W1(:,k+1),W2(k+1,:),b1(:,k+1),b2(k+1));
plot(p_vals,finalOutput)
plot(p_vals,G(p_vals))
hold off
xlabel('p')
ylabel('G(p)')
legend('Network output','Actual')

figure(2)
plot(error)
xlabel('iteration')
ylabel('mean square error')

%ptest_vals = (rand(1,20)-0.5)*4;
%Using the same 20 random test values every time
ptest_vals =     [ 0.0411;
   -0.4826;
    0.7391;
   -1.1862;
   -1.6137;
    1.6544;
   -1.3299;
    1.4611;
    1.7386;
    1.7941;
    1.1104;
   -1.3602;
   -1.3662;
    0.8122;
   -1.3203;
   -0.0271;
    0.5895;
    1.1679;
   -1.9711;
   -0.6896 ]';
finaltarget = double(G(ptest_vals));
finalnetoutput=myNet(ptest_vals, W1(:,k+1),W2(k+1,:),b1(:,k+1),b2(k+1));
finalerror = sum((finaltarget-finalnetoutput).^2)/length(finaltarget)   


function output = f1(a)
    output = (1-a)*a;
end

function output = f2()
    output = 1;
end

function output = myNet(p,W1,W2,b1,b2)
    a1=logsig(W1*p+b1);
    a2=W2*a1+b2;
    output = a2;
end
