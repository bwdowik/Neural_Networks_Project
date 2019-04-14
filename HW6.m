%% Steepest descent backpropogation algorithm
clear all

syms p;
syms G(p);
G(p) = 1 + sin(pi*p/2);
p_vals = linspace(-2,2,100);

plot(p_vals,G(p_vals))
xlabel('p')
ylabel('G(p)')

W_initial = rand(2,2)
b_initial = rand(2,1)



%% set up F







%%

%propagate input forward
alpha = 0.01;
p = 2;
M = 2;
W = ones(M+1,1);
b = ones(M+1,1);
t = 4;

difference = 1;
k=1;
while difference>0.000001

a(1)=p;
for m=1:M
    a(m+1) = logsig(W(m+1)*a(m)+b(m+1)); %logsig f needs to change depending on layer
end
a_final = a(M+1);

%propagate sensitivities backward
s(M+1) = -2*F_logsig(n(M+1))*(t-a_final); %only valid if sigmoid transfer
for m=M:-1:1
    s(m) = F_logsig(n(m))*W(m+1)'*s(m+1); %only valid if sigmoid transfer
end

%update weights and biases using steepest descent
W(m,k+1)=W(m,k) - alpha*s(m)*a(m-1)';
b(m,k+1) = b(m,k)-alpha*s(m);



    g(:,k)=Fgrad(x(1,k),x(2,k));
    p(:,k) = -g(:,k);
    A = double(Fhess(x(1,k),x(2,k)));
    a(k) = -double(g(:,k)'*p(:,k)/(p(:,k)'*A*p(:,k)))
    x(:,k+1)=x(:,k)-a(k)*g(:,k)
    difference = sqrt((x(1,k+1)-x(1,k))^2 + (x(2,k+1)-x(2,k))^2);
    k=k+1;
end


%% Functions

n = [1 4 6]
F_logsig(n)

function output = F_logsig(n)
    output=zeros(length(n));
    for q=1:length(n)
        output(q,q) =fsig(n(q));
    end
end

function output2 = fsig(x)
    output2 = logsig(x)*(1-logsig(x));
end

