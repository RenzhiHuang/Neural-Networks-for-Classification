% This function returns the predicted labels and error with a neural
% networks classifier.
function [error,y_hat] = ...
    neural_networks_classifier(testdata,testlabel,w1,b1,w2,b2,d1)
d = size(testdata,2);
m = size(testdata,1);
z = w1*testdata'+repmat(b1,1,m);% d1*m
a1 = sigmoid(z); % d1*m
f = w2 * a1+repmat(b2,1,m);% 1*m
eta = sigmoid(f);
eta(eta >1/2)=1;
eta(eta <=1/2)=0;
y_hat = eta;
acc = find(y_hat' == testlabel);
correct = size(acc,1);
error = 1-(1/size(testlabel,1)) * sum(correct);
end