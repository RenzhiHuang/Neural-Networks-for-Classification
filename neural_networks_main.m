% This code process data with a neural networks classifier.
d=57;
i=6;
d1=[1,5,10,15,25,50];
%get the train data and train label
train_data = train(:,1:d);
train_label = train(:,d+1);
train_label(train_label == -1) =0;
test_data = test(:,1:d);
test_label = test(:,d+1);
test_label(test_label == -1) =0;
% train error and test error
[w1_train,w2_train,b1_train,b2_train]=...
    neural_networks_train(train_data,train_label,w_1,b1,w_2,b2,d1(i));
train_error(i)=...
    neural_networks_classifier...
    (train_data,train_label,w1_train,b1_train,w2_train,b2_train,d1(i));
test_error(i)=...
    neural_networks_classifier...
    (test_data,test_label,w1_train,b1_train,w2_train,b2_train,d1(i));

%% Cross validation
sum_test_error = 0;
for fold=1:5
    cvtrain_data = cv_sub_train{fold}(:,1:d);
    cvtrain_label = cv_sub_train{fold}(:,d+1);
    cvtrain_label(cvtrain_label == -1) =0;
    cvtest_data = cv_sub_test{fold}(:,1:d);
    cvtest_label = cv_sub_test{fold}(:,d+1);
    cvtest_label(cvtest_label == -1) =0;
    [w1_cvtrain,w2_cvtrain,b1_cvtrain,b2_cvtrain]=...
        neural_networks_train(cvtrain_data,cvtrain_label,w_1,b1,w_2,b2,d1(i));
    sum_test_error = sum_test_error+...
        neural_networks_classifier...
        (cvtest_data,cvtest_label,w1_cvtrain,b1_cvtrain,w2_cvtrain,b2_cvtrain,d1(i));
end
cv_error(i) = sum_test_error/5;