clc;
clear;

% Process training data -> add mask path
% Process test data     -> uncomment 'masks_tif = zeros(size(slices_tif));'
%train_test_data_slices = '../training_data/slices/';
%train_test_data_masks = '../training_data/masks/';

train_test_data_slices = '../test_data/slices/';
train_test_data_masks = '../test_data/masks/';

slices_nii_folder=dir(train_test_data_slices);
slices_nii_file={slices_nii_folder.name};

% Traverse all .nii.gz file
for num_nii = 4 : length(slices_nii_file)
    
    case_name = slices_nii_file(num_nii);
    case_name = char(case_name);
    case_name = case_name(1 : end-7);
    
    finishing = [num2str(num_nii-3),'/',num2str(length(slices_nii_file)-3)];
     disp(finishing)
     disp(case_name)
    
    v_slices = load_untouch_nii([train_test_data_slices, case_name, '.nii.gz']);  
    %v_masks = load_untouch_nii([train_test_data_masks, case_name, '.manual.mask.nii.gz']);
    v_masks = load_untouch_nii([train_test_data_masks, case_name, '_ss.nii.gz']);
    slices_tif = v_slices.img;
    masks_tif = v_masks.img;
    
    %masks_tif = zeros(size(slices_tif));
    
    [n1,n2,n3] = size(slices_tif);
    

%% Save as tiff
    for i = 1 : n3 
        if i == 1
            slices = im2uint8(rescale(slices_tif(:,:,1), 0, 1));
            masks = im2uint8(rescale(masks_tif(:,:,1), 0, 1));
        else
            single_slice = im2uint8(rescale(slices_tif(:,:,i), 0, 1));
            single_mask = im2uint8(rescale(masks_tif(:,:,i), 0, 1));
            slices = cat(3, slices, single_slice); 
            masks = cat(3, masks, single_mask);
        end
    end
    
%% 
    %destination_path = './data/test_model/';    % Get qualitative results
    % destination_path = './data/valid/'; 
    %destination_path = './data/train/'; 
    destination_path = ['./data/test/', case_name, '/'];
     
%% classify into two categories    
    if max(max(max(slices_tif))) > 1220
        [slices_preprocessed, mask_preprocessed] = preprocessing_high(slices, masks, destination_path, case_name, n1, n2, n3);
    else       
        [slices_preprocessed, mask_preprocessed] = preprocessing(slices_tif, masks, destination_path, case_name, n1, n2, n3);  
    end 
end


%% functions
function [ slices, mask] = preprocessing_high(slices, mask, destination_path, prefix, n1, n2, n3 )      
    save_preprocessed_images(slices, mask, destination_path, prefix, n1, n2, n3);
end

function [slices, mask] = preprocessing(slices, mask, destination_path, prefix, n1, n2, n3 )      
    slices = double(slices);
    slices = rescale(slices, 0, 255);
    % get histogram of an image volume
    [N, edges] = histcounts(slices(:), 'BinWidth', 2);

    % rescale the intensity peak to be at value 100
    minimum = edges(find(edges > prctile(slices(:), 2), 1));

    diffN = zeros(size(N));
    for nn = 2:numel(N)
        diffN(nn) = N(nn) / N(nn - 1);
    end
    s = find(edges >= prctile(slices(:), 50), 1);
    f = find(diffN(s:end) > 1.0, 5);
    start = s + f(5);

    [~, ind] = max(N(start:end));
    peak_val = edges(ind + start - 1);
    maximum = minimum + ((peak_val - minimum) * 2.55);

    slices(slices < minimum) = minimum;
    slices(slices > maximum) = maximum;
    slices = (slices - minimum) ./ (maximum - minimum);
    save_preprocessed_images(slices, mask, destination_path, prefix, n1, n2, n3);
end



function [] = save_preprocessed_images(slices, mask, destination_path, prefix, n1, n2, n3)

% save preprocessed images
    slices = im2uint8(slices);
    mask = 255 * (mask);
 % center crop to 256x256 square
    slices = center_crop(slices, n1, n2, n3);
    mask = center_crop(mask, n1, n2, n3);   
    
    
    slicesPerImage = 1;
    easy_sort = 10000;
    for s = size(slices, 3):-slicesPerImage:1
        startSlice = max([1, (s - slicesPerImage + 1)]);
        imageSlices = slices(:, :, startSlice:(startSlice + slicesPerImage - 1));
        maskSlices = mask(:, :, startSlice:(startSlice + slicesPerImage - 1));
        
        figure(1)
        imshow(imageSlices)
        
        %%% save vaild image for training 
        %if (startSlice) > 45 && (startSlice) < n3-45
            saveastiff(imageSlices, [destination_path prefix '_' num2str(easy_sort + startSlice) '.tif']);
            saveastiff(maskSlices, [destination_path prefix '_' num2str(easy_sort + startSlice) '_mask.tif']);
        %end
        
        
    end

end


function [ image ] = center_crop(image, n1, n2, n3)
    num_pad_n1 = 256-n1;
    num_pad_n2 = 256-n2; 
    
    num_pad_n1_half = round(num_pad_n1/2);
    num_pad_n2_half = round(num_pad_n2/2);
    
    if and(n1<256, n2<256)
        image = padarray(image, [num_pad_n1_half, 0], 'pre');
        image = padarray(image, [num_pad_n1 - num_pad_n1_half, 0 ], 'post');
        image = padarray(image, [0, num_pad_n2_half], 'pre');  
        image = padarray(image, [0, num_pad_n2 - num_pad_n2_half], 'post');
        
    end
    
    if and(n1<256, n2>=256)
        image = padarray(image, [num_pad_n1_half, 0 ], 'pre');
        image = padarray(image, [num_pad_n1 - num_pad_n1_half, 0 ], 'post');  
    end
    
    if and(n1>=256, n2<256)
        image = padarray(image, [0, num_pad_n2_half], 'pre');  
        image = padarray(image, [0, num_pad_n2 - num_pad_n2_half], 'post');  
    end

    image_size = size(image);
    if or(image_size(1)>256, image_size(2)>256)
        win1 = centerCropWindow3d(size(image), [256,256, n3]);
        image = imcrop3(image, win1);  
    end
end

