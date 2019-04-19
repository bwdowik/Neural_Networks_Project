%% HW6_part 1, SDBP

clear all
clc
clf

syms p;
syms G(p);
G(p) = 1 + sin(pi*p/4);
p_vals = linspace(-2,2,100);
batch_size=5;
alpha = 0.05*batch_size;
k=1;
target = double(G(p_vals));
tau=0.618;
epsilon = 0.1;

%random initial weights, will change each time, might want to not use for testing
% W1(:,k) = [rand(1)-0.5; rand(1)-0.5];
% b1(:,k) = [rand(1)-0.5; rand(1)-0.5];
% W2(k,:) = [rand(1)-0.5 rand(1)-0.5];
% b2(k) = [rand(1)-0.5];

% might be better to use these for testing
W1(:,k) = [-0.27; -0.41];
b1(:,k) = [-0.48; -0.13];
W2(k,:) = [0.09 -0.17];
b2(k) = [0.48];

%trying to modify for batching
index=1;
rand_index = randperm(length(p_vals));

for k=1:50
    W2sum=0;
    b2sum=0;
    W1sum=0;
    b1sum=0;
    
    for q=1:batch_size
        %This stuff is so we just train it with randomly ordered values
        p = p_vals(rand_index(index));
        index=index+1;
        if index>=100
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
    

    W2loop(1,:) = W2(k,:);
    b2loop(1) = b2(k);
    W1loop(:,1) = W1(:,k);
    b1loop(:,1)=b1(:,k);
    netoutput=myNet(p_vals, W1loop(:,1),W2loop(1,:),b1loop(:,1),b2loop(1));
    errorloop(1) = sum((target-netoutput).^2)/length(target); 
    
    %interval location step
    mult=1;
    r=2;
    decreasing=1;
    while decreasing==1
        W2loop(r,:) = W2loop(r-1,:)-mult*alpha*W2sum/batch_size;
        b2loop(r) = b2loop(r-1)-mult*alpha*b2sum/batch_size;
        W1loop(:,r) = W1loop(:,r-1)-mult*alpha*W1sum/batch_size;
        b1loop(:,r)=b1loop(:,r-1)-mult*alpha*b1sum/batch_size;
        mult = mult*2;
        
        %Check the performance index at this point
        netoutput=myNet(p_vals, W1loop(:,r),W2loop(r,:),b1loop(:,r),b2loop(r));
        errorloop(r) = sum((target-netoutput).^2)/length(target);   
        
        if(errorloop(r)>errorloop(r-1))
            decreasing = 0;
        end
        
        r=r+1;
    end
    
    if r>3
        %Interval reduction using Golden Section search
        %We know the minimum is between points with index r-3 and r-1
        W2a(1,:) = W2loop(r-3,:);
        b2a(1) = b2loop(r-3);
        W1a(:,1) = W1loop(:,r-3);
        b1a(:,1)=b1loop(:,r-3);    

        W2b(1,:) = W2loop(r-1,:);
        b2b(1) = b2loop(r-1);
        W1b(:,1) = W1loop(:,r-1);
        b1b(:,1)=b1loop(:,r-1);

        W2c(1,:) = W2a(1,:)+(1-tau)*(W2b(1,:)-W2a(1,:));
        b2c(1) = b2a(1)+(1-tau)*(b2b(1)-b2a(1));
        W1c(:,1) = W1a(:,1)+(1-tau)*(W1b(:,1)-W1a(:,1));
        b1c(:,1)=b1a(:,1)+(1-tau)*(b1b(:,1)-b1a(:,1));
        netoutputc=myNet(p_vals, W1c(:,1),W2c(1,:),b1c(:,1),b2c(1));
        Fc = sum((target-netoutputc).^2)/length(target);

        W2d(1,:) = W2b(1,:)-(1-tau)*(W2b(1,:)-W2a(1,:));
        b2d(1) = b2b(1)-(1-tau)*(b2b(1)-b2a(1));
        W1d(:,1) = W1b(:,1)-(1-tau)*(W1b(:,1)-W1a(:,1));
        b1d(:,1)=b1b(:,1)-(1-tau)*(b1b(:,1)-b1a(:,1));
        netoutputd=myNet(p_vals, W1d(:,1),W2d(1,:),b1d(:,1),b2d(1));
        Fd = sum((target-netoutputd).^2)/length(target);

        z=1;
        tol = 0.0005;
        a(z)=sqrt(sum(W2a(z,:).^2)+sum(W1a(:,z).^2)+sum(b1a(:,z).^2)+sum(b2a(z).^2));
        b(z)=sqrt(sum(W2b(z,:).^2)+sum(W1b(:,z).^2)+sum(b1b(:,z).^2)+sum(b2b(z).^2));
        
        while abs(b(z)-a(z))>tol
            if Fc<Fd
                W2a(z+1,:)=W2a(z,:);
                b2a(z+1)=b2a(z);
                W1a(:,z+1)=W1a(:,z);
                b1a(:,z+1)=b1a(:,z);
                
                W2b(z+1,:)=W2d(z,:);
                b2b(z+1)=b2d(z);
                W1b(:,z+1)=W1d(:,z);
                b1b(:,z+1)=b1d(:,z);
                
                W2d(z+1,:)=W2c(z,:);
                b2d(z+1)=b2c(z);
                W1d(:,z+1)=W1c(:,z);
                b1d(:,z+1)=b1c(:,z);
                
                W2c(z+1,:) = W2a(z+1,:)+(1-tau)*(W2b(z+1,:)-W2a(z+1,:));
                b2c(z+1) = b2a(z+1)+(1-tau)*(b2b(z+1)-b2a(z+1));
                W1c(:,z+1) = W1a(:,z+1)+(1-tau)*(W1b(:,z+1)-W1a(:,z+1));
                b1c(:,z+1)=b1a(:,z+1)+(1-tau)*(b1b(:,z+1)-b1a(:,z+1));
                
                Fd = Fc;
                netoutputc=myNet(p_vals, W1c(:,z+1),W2c(z+1,:),b1c(:,z+1),b2c(z+1));
                Fc = sum((target-netoutputc).^2)/length(target);
            else %if Fd>Fc
                W2a(z+1,:)=W2c(z,:);
                b2a(z+1)=b2c(z);
                W1a(:,z+1)=W1c(:,z);
                b1a(:,z+1)=b1c(:,z);
                
                W2b(z+1,:)=W2b(z,:);
                b2b(z+1)=b2b(z);
                W1b(:,z+1)=W1b(:,z);
                b1b(:,z+1)=b1b(:,z);
                
                W2c(z+1,:)=W2d(z,:);
                b2c(z+1)=b2d(z);
                W1c(:,z+1)=W1d(:,z);
                b1c(:,z+1)=b1d(:,z);
                
                W2d(z+1,:) = W2b(z+1,:)-(1-tau)*(W2b(z+1,:)-W2a(z+1,:));
                b2d(z+1) = b2b(z+1)-(1-tau)*(b2b(z+1)-b2a(z+1));
                W1d(:,z+1) = W1b(:,z+1)-(1-tau)*(W1b(:,z+1)-W1a(:,z+1));
                b1d(:,z+1)=b1b(:,z+1)-(1-tau)*(b1b(:,z+1)-b1a(:,z+1));
                
                Fc = Fd;
                netoutputd=myNet(p_vals, W1d(:,z+1),W2d(z+1,:),b1d(:,z+1),b2d(z+1));
                Fd = sum((target-netoutputd).^2)/length(target);          
            end
            
            a(z+1)=sqrt(sum(W2a(z+1,:).^2)+sum(W1a(:,z+1).^2)+sum(b1a(:,z+1).^2)+sum(b2a(z+1).^2));
            b(z+1)=sqrt(sum(W2b(z+1,:).^2)+sum(W1b(:,z+1).^2)+sum(b1b(:,z+1).^2)+sum(b2b(z+1).^2));
            z=z+1;
        end
        
        W2(k+1,:) = W2b(z,:);
        b2(k+1) = b2b(z);
        W1(:,k+1) = W1b(:,z);
        b1(:,k+1)=b1b(:,z);    
    
    else %if the first value (a) was lower than the second value
        W2(k+1,:) = W2loop(r-2,:);
        b2(k+1) = b2loop(r-2);
        W1(:,k+1) = W1loop(:,r-2);
        b1(:,k+1)=b1loop(:,r-2);
    end
    
    
    netoutput=myNet(p_vals, W1(:,k+1),W2(k+1,:),b1(:,k+1),b2(k+1));
    error(k) = sum((target-netoutput).^2)/length(target);   
end

%% 
figure(1)
hold on
plot(p_vals,myNet(p_vals, W1(:,k+1),W2(k+1,:),b1(:,k+1),b2(k+1)))
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
ptest_vals =     [ 0.0411;-0.4826;0.7391;-1.1862;-1.6137;1.6544;-1.3299;
    1.4611;1.7386;1.7941;1.1104;-1.3602;-1.3662;0.8122;-1.3203;-0.0271;
    0.5895;1.1679;-1.9711;-0.6896 ]';
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
