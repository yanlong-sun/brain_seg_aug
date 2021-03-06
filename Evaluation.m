clc;
clear;
%% Evaluate deep learning model
prediction_path = '../predictions_unet/';
pred_folder= dir(prediction_path);
pred_file={pred_folder.name};
dice_coef_dl = zeros(1, length(pred_file)-3);
case_name_list = string(pred_file(4:length(pred_file)));
for num_pred= 4 : length(pred_file)
    case_name = pred_file(num_pred);
    case_name = char(case_name);
    mat = load([prediction_path, case_name, '/predictions.mat']);
    mask = logical(mat.mask);
    pred = logical(mat.pred);   
    dice =  2*nnz(mask&pred)/(nnz(mask) + nnz(pred));
    dice_coef_dl(num_pred-3) = dice;

end
    dice_dl_avg = mean(dice_coef_dl)

    
% Evaluate brainsuite
prediction_path = '../predictions_brainsuite/';
masks_path = '../test_data/masks/';
pred_folder= dir(prediction_path);
pred_file={pred_folder.name};
dice_coef_bse = zeros(1, length(pred_file)-3);
for num_pred= 4 : length(pred_file)
    case_name = pred_file(num_pred);
    case_name = char(case_name);   
    preds_nii = load_untouch_nii([prediction_path, case_name, '/',case_name, '.mask.nii.gz']);  
    masks_nii = load_untouch_nii([masks_path, case_name, '_ss.nii.gz']);
    pred = logical(preds_nii.img);
    mask = logical(masks_nii.img);
    dice =  2*nnz(mask&pred)/(nnz(mask) + nnz(pred));
    dice_coef_bse(num_pred-3) = dice;
end
    dice_bse_avg = mean(dice_coef_bse)


%% plot
figure(1)
x = cat(2, dice_coef_dl', dice_coef_bse');
y = categorical(case_name_list);
barh(y, x)
set(gca,'FontSize',9);
xlabel('Dice Coefficient')
xlim([0.7, 1])
ylabel('Case Name')
xlabel('Dice Coefficient');
grid on;
ax = gca;
ax.LineWidth = 2;
ylim=get(gca,'Ylim');
line([dice_dl_avg, dice_dl_avg], ylim, 'Color','blue','LineStyle','--', 'LineWidth',2 );
line([dice_bse_avg, dice_bse_avg], ylim, 'Color','red','LineStyle','--', 'LineWidth',2 );
text(dice_dl_avg, ylim(2), num2str(dice_dl_avg), 'color','b');
text(dice_bse_avg, ylim(2), num2str(dice_bse_avg), 'color','r');
legend({'U-Net', 'Brainsuite', 'Avg DSC for UNet','Avg DSC for brainsuite'});

