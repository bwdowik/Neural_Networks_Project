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

alpha = 0.01;

