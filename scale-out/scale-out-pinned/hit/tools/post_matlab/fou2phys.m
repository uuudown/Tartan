function res = fou2phys(u)
  % Doubles the spectrum and inverse fft. Understand the what, not the how.
  res = ifft(u,[],3);
  res = ifft(res,[],2);
  res = real(ifft(cat(1,res(1:end-1,:,:),conj(res(end:-1:2,:,:)))));
