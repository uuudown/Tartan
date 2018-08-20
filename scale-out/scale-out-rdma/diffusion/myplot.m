function myplot(result,nx,ny,nz,L,W,H,name)

[x,y,z]=meshgrid(linspace(-L/2,L/2,nx),linspace(-W/2,W/2,ny),linspace(-H/2,H/2,nz));
u = reshape(result(:,1),nx,ny,nz);
v = reshape(result(:,2),nx,ny,nz);
w = reshape(result(:,3),nx,ny,nz);
p = reshape(result(:,4),nx,ny,nz);

% Plots results in Matlab
figure; 
subplot(2,2,1); h1=slice(x,y,z,p,0,0,[]); 
cm=getColorMap('kwave'); colormap(cm); axis tight; colorbar; %caxis([0,4]);
h1(1).EdgeColor = 'none'; xlabel('x'); set(gca,'XTick',[-0.02,-0.01,0,0.01,0.02]);
h1(2).EdgeColor = 'none'; ylabel('y'); set(gca,'YTick',[-0.02,-0.01,0,0.01,0.02]);
%h1(3).EdgeColor = 'none'; 
zlabel('z'); set(gca,'ZTick',[-0.02,-0.01,0,0.01,0.02]);
title('SSP-RK3+WENO5 pressure.');
subplot(2,2,2); h2=slice(x,y,z,u,0,0,[]);
cm=getColorMap('kwave'); colormap(cm); axis tight; colorbar; %caxis([0,4]);
h2(1).EdgeColor = 'none'; xlabel('x'); set(gca,'XTick',[-0.02,-0.01,0,0.01,0.02]);
h2(2).EdgeColor = 'none'; ylabel('y'); set(gca,'YTick',[-0.02,-0.01,0,0.01,0.02]);
%h2(3).EdgeColor = 'none'; zlabel('z'); 
set(gca,'ZTick',[-0.02,-0.01,0,0.01,0.02]);
title('SSP-RK3+WENO5 x-velocity.');
subplot(2,2,3); h3=slice(x,y,z,v,0,0,[]);
cm=getColorMap('kwave'); colormap(cm); axis tight; colorbar; %caxis([0,4]);
h3(1).EdgeColor = 'none'; xlabel('x'); set(gca,'XTick',[-0.02,-0.01,0,0.01,0.02]);
h3(2).EdgeColor = 'none'; ylabel('y'); set(gca,'YTick',[-0.02,-0.01,0,0.01,0.02]);
%h3(3).EdgeColor = 'none'; zlabel('z'); 
set(gca,'ZTick',[-0.02,-0.01,0,0.01,0.02]);
title('SSP-RK3+WENO5 z-velocity.');
subplot(2,2,4); h4=slice(x,y,z,w,0,0,[]);
cm=getColorMap('kwave'); colormap(cm); axis tight; colorbar; %caxis([0,4]);
h4(1).EdgeColor = 'none'; xlabel('x'); set(gca,'XTick',[-0.02,-0.01,0,0.01,0.02]);
h4(2).EdgeColor = 'none'; ylabel('y'); set(gca,'YTick',[-0.02,-0.01,0,0.01,0.02]);
%h4(3).EdgeColor = 'none'; zlabel('z'); 
set(gca,'ZTick',[-0.02,-0.01,0,0.01,0.02]);
title('SSP-RK3+WENO5 z-velocity.');

print([name,'.png'],'-dpng')

% Export Result for Paraview

% Open the file.
fid = fopen([name,'.vtk'], 'w');
if fid == -1
    error('Cannot open file for writing.');
end
fprintf(fid,'# vtk DataFile Version 2.0\n');
fprintf(fid,'Volume example\n');
fprintf(fid,'BINARY\n');
fprintf(fid,'DATASET STRUCTURED_POINTS\n');
fprintf(fid,'DIMENSIONS %d %d %d\n',nx,ny,nz);
fprintf(fid,'ASPECT_RATIO %d %d %d\n',1,1,1);
fprintf(fid,'ORIGIN %d %d %d\n',0,0,0);
fprintf(fid,'POINT_DATA %d\n',nx*ny*nz);
fprintf(fid,'SCALARS Pressure float 1\n');
fprintf(fid,'LOOKUP_TABLE default\n');
fwrite(fid,result(:,4),'float','ieee-be');
