% This function trains a neural networks classifier and returns all the
% weights and bias.
function [w1,w2,b1,b2]=...
    neural_networks_train(traindata,trainlabel,w1,b1,w2,b2,d1)
d = size(traindata,2);
m = size(traindata,1);
%x(:,d+1)=ones(m,1);
w1 = w1';%d1*d
w2 = w2';
b1 = b1';%d1*1
% total iterations = 5000
for i = 1:5000
    z = w1*traindata'+repmat(b1,1,m);% d1*m
    a1 = sigmoid(z); % d1*m
    f = w2 * a1+repmat(b2,1,m);% 1*m
    deriv_w2 = (sigmoid(f)-trainlabel')*a1';%1*d1
    deriv_w1 = (sigmoid(f)-trainlabel').*(w2'.*deriv_sigmoid(z)) * traindata;%d1*d
    deriv_b2 = sum(sigmoid(f)-trainlabel');%scalar
    deriv_b1 = (sigmoid(f)-trainlabel')*(w2'.*deriv_sigmoid(z))';%1*d1
    %update w2 an b2
    for j=1:d1
        w2(1,j)=w2(1,j)-0.1 / m * deriv_w2(j);
    end
    b2 = b2 -0.1/m * deriv_b2;
    %update w1
    for j=1:d1
        for k=1:d
            w1(j,k) = w1(j,k)-0.1 / m * deriv_w1(j,k);
        end
        b1(j)=b1(j)-0.1/m*deriv_b1(j);
    end
end
end

        
    
