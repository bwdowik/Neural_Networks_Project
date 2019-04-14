%% HW6_part 1

clear all
clc
clf

syms p;
syms G(p);
G(p) = 1 + sin(pi*p/2);
p_vals = linspace(-2,2,200);
alpha = 0.1;
batch_size=5;
k=1;
W1(:,k) = [-0.27; -0.41];
b1(:,k) = [-0.48; -0.13];
W2(k,:) = [0.09 -0.17];
b2(k) = [0.48];

%trying to modify for batching
% for k=1:1000
%     for q=1:batch_size
%         p = p_vals(randi(200,1,1));
%         t(q) = double(G(p));
%         a0=p;
% 
% 
%         a1(q,:)=logsig(W1(:,k)*a0+b1(:,k));
%         a2(q)=W2(k,:)*a1+b2(k);
%         
%         e = t-a2;
% 
%         s2(q) = -2*f2()*e;
% 
%         F1n1 = zeros(2);
%         F1n1(1,1) = f1(a1(1));
%         F1n1(2,2) = f1(a1(2));
%         s1(q) = F1n1*W2(k,:)'*s2;       
%     end
%     
%     W2(k+1,:) = W2(k,:)-alpha*s2*a1';
%     b2(k+1) = b2(k)-alpha*s2;
%     W1(:,k+1) = W1(:,k)-alpha*s1*a0';
%     b1(:,k+1)=b1(:,k)-alpha*s1;
%     
% end

% the one that works, no batching
for k=1:1000
    
    p = p_vals(randi(200,1,1));
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
    
    W2(k+1,:) = W2(k,:)-alpha*s2*a1';
    b2(k+1) = b2(k)-alpha*s2;
    W1(:,k+1) = W1(:,k)-alpha*s1*a0';
    b1(:,k+1)=b1(:,k)-alpha*s1;
    
end


figure(1)
hold on
plot(p_vals,myNet(p_vals, W1(:,k+1),W2(k+1,:),b1(:,k+1),b2(k+1)))
plot(p_vals,G(p_vals))
hold off
xlabel('p')
ylabel('G(p)')
legend('Network output','Actual')

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