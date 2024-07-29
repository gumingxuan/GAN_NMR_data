clc;clear;close all

r = corrcoef(pre_RF,time_opt);r = r(1, 2); R2(1) = r^2;
r = corrcoef(pre_SVM,time_opt);r = r(1, 2); R2(2) = r^2;
r = corrcoef(pre_ANN,time_opt);r = r(1, 2); R2(3) = r^2;

error1(1,1)=mean(abs(Sw_cal_SINGLE-Sw_real)./Sw_real);
error1(1,2)=mean(abs(Sw_cal_RF-Sw_real)./Sw_real);
error1(1,3)=mean(abs(Sw_cal_SVM-Sw_real)./Sw_real);
error1(1,4)=mean(abs(Sw_cal_ANN-Sw_real)./Sw_real);

error2(1,1)=(Sw_cal_SINGLE-Sw_real)'*(Sw_cal_SINGLE-Sw_real)/500;
error2(1,2)=(Sw_cal_RF-Sw_real)'*(Sw_cal_RF-Sw_real)/500;
error2(1,3)=(Sw_cal_SVM-Sw_real)'*(Sw_cal_SVM-Sw_real)/500;
error2(1,4)=(Sw_cal_ANN-Sw_real)'*(Sw_cal_ANN-Sw_real)/500;

error3(1,1)=(T22(single_cutoff)-T22(mean_single_cutoff))*(T22(single_cutoff)-T22(mean_single_cutoff))'/500;
error3(1,2)=(time_opt-pre_RF)'*(time_opt-pre_RF)/500;
error3(1,3)=(time_opt-pre_SVM)'*(time_opt-pre_SVM)/500;
error3(1,4)=(time_opt-pre_ANN)'*(time_opt-pre_ANN)/500;





    
