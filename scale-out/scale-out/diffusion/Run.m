% Produce Run
close all;

% =======================Diffusion-3D MPI-GPU-FD4=====================
% Optimization                                 :  Pitched Memory
% Kernel time ex. data transfers               :  6.802646 seconds
% Data transfer(s) HtD                         :  0.044401 seconds
% Data transfer(s) DtH                         :  0.041844 seconds
% ===================================================================
% Total effective GFLOPs                       :  5.872351
% ===================================================================
% 3D Grid Size                                 :  400 x 200 x 206
% Iterations                                   :  101 x 3 RK steps
% ===================================================================

% Set run parameters
K = 1.0;
L = 2.0;
W = 2.0;
H = 2.0;
nx = 400;
ny = 200;
nz = 200;
iter = 1000;
block_X = 64;
block_Y = 4;
block_Z = 1;
np = 2;
RADIUS = 3;

% Write sh.run
fID = fopen('run.sh','wt');
fprintf(fID,'make\n');
args = sprintf('%1.2f %1.2f %1.2f %1.2f %d %d %d %d %d %d %d',K,L,W,H,nx,ny,nz,iter,block_X,block_Y,block_Z);
%fprintf(fID,'mpirun -np %d Diffusion3d.run %s\n',np,args); 
profile = 'nvprof -f -o Diffusion3d.%q{OMPI_COMM_WORLD_RANK}.nvprof';
fprintf(fID,'mpirun -np %d %s ./Diffusion3d.run %s\n',np,profile,args);
fclose(fID);

% Execute sh.run
! sh run.sh

% Build discrete domain
dx=L/(nx-1); xc=-L/2:dx:L/2;
dy=W/(ny-1); yc=-W/2:dy:W/2;
dz=H/(nz-1); nz=nz+2*RADIUS; zc=-H/2-RADIUS*dz:dz:H/2+RADIUS*dz;
[x,y,z]=meshgrid(yc,xc,zc);

% Set plot region
region = [-L/2,L/2,-W/2,W/2,-H/2,H/2]; 

% Load and plot data
fID = fopen('result.bin');
output = fread(fID,[1,nx*ny*nz],'float')';
u = reshape(output,nx,ny,nz);

%% myplot(output,nx,ny,nz,L,W,H,'result');
figure(1)
q=slice(x,y,z,u,0,0,0); axis(region);
title('Heat Equation, MultiGPU-FDM-RK3','interpreter','latex','FontSize',18);
q(1).EdgeColor = 'none';
q(2).EdgeColor = 'none';
q(3).EdgeColor = 'none';
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
zlabel('$\it{z}$','interpreter','latex','FontSize',14);
colorbar;
print('InviscidBurgers_MPI_CUDA_3d','-dpng');

%% Load and plot Initial Condition
fID = fopen('initial.bin');
output = fread(fID,[1,nx*ny*nz],'float')';
u0 = reshape(output,nx,ny,nz);

%% myplot(output,nx,ny,nz,L,W,H,'result');
figure(2)
q=slice(x,y,z,u0,0,0,0); axis(region);
title('Initial Condition','interpreter','latex','FontSize',18);
q(1).EdgeColor = 'none';
q(2).EdgeColor = 'none';
q(3).EdgeColor = 'none';
xlabel('$\it{x}$','interpreter','latex','FontSize',14);
ylabel('$\it{y}$','interpreter','latex','FontSize',14);
zlabel('$\it{z}$','interpreter','latex','FontSize',14);
colorbar;
print('InitialCondition_MPI_CUDA_3d','-dpng');

% Clean up
! rm -rf *.bin 
