function res = readfield(filename)
% Reads HIT in hdf5 and returns the usual spectrum
% Requires a fairly up to date version of Matlab or Octave

% Check if octave
  isOctave = exist('OCTAVE_VERSION') ~= 0;
  if isOctave
    u = load('-hdf5',filename);
    u = u.u;
  else
    u = hdf5read(filename,'u');
  end

  sizeu = size(u);
  res = complex(zeros(sizeu(1)/2,sizeu(2),sizeu(3)),0);
  res(:,:,:) = u(1:2:end-1,:,:) + 1i.*u(2:2:end,:,:);
