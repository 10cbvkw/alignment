% example of use to register a 2D slice to a 3D volume
clear all
close all

%导入体数据
volumdata = load('D:\SliVo-master\data\recondata_cut.mat');
volumdata = volumdata.recondata_cut;
volumdata = permute(volumdata,[2,3,1]);
%导入片数据
slicedata = imread('D:\SliVo-master\data\20220605-photoacoustic-CA1-3-Orthogonal Projection-10_c1-2.png');
% slicedata = slicedata(:,365:8100,:);
slicedata_b = slicedata(:,:,1);
slicedata_r = slicedata(:,:,3);
figure
imshow(slicedata_r);
% figure
% for i = 1:250
% imshow(volumdata(:,:,i))
% pause(0.05)
% end
slicedatainput = imresize(slicedata_r,0.1);
slicedatainput = rot90(slicedatainput);
slicedatainput = rot90(slicedatainput);
slicedatainput = flipdim(slicedatainput,2);


slicedata_b = imresize(slicedata_b,0.1);
slicedata_b = rot90(slicedata_b);
slicedata_b = rot90(slicedata_b);

figure
imshowpair(slicedatainput,volumdata(:,:,100),'montage')
% imshow(slicedatainput);
% imshow(volumdata(:,:,i))

%导入光声探针
Data3Dc = load('D:\SliVo-master\data\fluorescenceCA1.mat');
Data3Dc = Data3Dc.fl;
Data3Dcut = permute(Data3Dc,[2,3,1]);

%导入荧光探针
slicedata_b = imresize(slicedata_b,0.1);
slicedata_b = rot90(slicedata_b);
slicedata_b = rot90(slicedata_b);
slicedata_b = flipdim(slicedata_b,2);

registerSliceToVolume(volumdata,slicedatainput,Data3Dcut,slicedata_b,'lower_limit',100, 'upper_limit', 197,'calculate_features', 1);



