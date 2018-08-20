function dhatdy = ddy(u)
  ssize = size(u);
  N = ssize(3);
  
  kz = (0:N/2)'+0*1i;
  ky = cat(1,kz(1:end-1),-kz(end:-1:2));
  ky = reshape(ky,1,N,1);
  dhatdy = zeros(N/2+1,N,N);
  
  for i = 1:N
    for k = 1:N/2+1
      dhatdy(k,:,i) = 1i*ky.*u(k,:,i);
    end
  end
