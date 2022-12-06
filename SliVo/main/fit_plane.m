function Slice_ransac = fit_plane(Data3D,Slice2D,options,Data3Dcut,slicedata_b)
% %导入光声探针
% Data3Dc = load('D:\PA\data0609\padata\fluorescenceCA1.mat');
% Data3Dc = Data3Dc.fl;
% Data3Dcut = permute(Data3Dc,[2,3,1]);
% %Data3Dcut = Data3Dc(:,:,100:197);
% % figure
% % for i=1:5:90
% % imshow(Data3Dcut(:,:,i),[0.8,1])
% % pause(0.06)
% % end

% load coordinates
load([options.folder_matches 'FeatureCoordinates_3D.mat']);
[X_size Y_size Z_size] = size(Data3D);

% calculate normal vector
[B,~]=matching_plane(FeatureCoordinates_3D, options);
save([options.folder_matches 'Normal_vec_toPlane.mat'],'B','-mat')

% show cut image
[x_mesh, y_mesh]=meshgrid(1:X_size,1:Y_size);
Z=-(y_mesh.*B(1) + x_mesh.*B(2) + B(4))/B(3);

Slice_ransac = interp3(single(Data3D),x_mesh,y_mesh,Z); %cut a slice from 3D volume
imwrite(uint8(Slice_ransac), [options.folder_matches 'FoundMatch_in_3D.tif'],'Compression','None' )

Slice_ransac2 = interp3(single(Data3Dcut),x_mesh,y_mesh,Z); %cut a slice from 3D volume
imwrite(uint8(Slice_ransac2), [options.folder_matches 'FoundMatch_in_3D2.tif'],'Compression','None' )

% slicedata = imread('D:\PA\data0609\fluorecence\20220605-photoacoustic-CA1-3-Orthogonal Projection-10_c1-2.png');
% % slicedata = slicedata(:,365:8100,:);
% slicedata_b = slicedata(:,:,1);
% slicedata_b = imresize(slicedata_b,0.1);
% slicedata_b = rot90(slicedata_b);
% slicedata_b = rot90(slicedata_b);
% slicedata_b = flipdim(slicedata_b,2);
% figure
% imshow(slicedata_b);

save('D:\PA\data0609\CA1results\slice-CA1-3-10_c1-2.mat','Slice_ransac');
save('D:\PA\data0609\CA1results\fprobe-CA1-3-10_c1-2.mat','Slice_ransac2');
save('D:\PA\data0609\CA1results\floreslice-CA1-3-10_c1-2.mat','Slice2D');
save('D:\PA\data0609\CA1results\floreprobe-CA1-3-10_c1-2.mat','slicedata_b');


figure,
subplot(2,2,1), imshow(Slice_ransac,[]), title('Found slice in 3D volume')
subplot(2,2,2), imshow(Slice_ransac2), title('Found slice in corr slice')
subplot(2,2,3), imshow(Slice2D), title('Given 2D slice')
subplot(2,2,4), imshow(slicedata_b,[]), title('Given corr slice')


