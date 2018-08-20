function dhatdz = ddz(u)

  ssize = size(u);
  N = ssize(3);

  kz = (0:N/2)'+0*1i;
  kz = reshape(kz,N/2+1,1,1);
  dhatdz = zeros(N/2+1,N,N);

  for i = 1:N
    for j = 1:N
      dhatdz(:,j,i) = 1i*kz.*u(:,j,i);
    end
  end
