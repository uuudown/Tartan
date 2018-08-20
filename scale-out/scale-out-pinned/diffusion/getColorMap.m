function cm = getColorMap(varargin)
%GETCOLORMAP    Return pressure field and flow fields color map.
%
% USAGE:
%       cm = getColorMap()          default option: jetvar
%       cm = getColorMap('cmap')    cmap: kwave, +pressure,-pressure, cold
%
% OPTIONAL INPUTS:
%       num_colors  - number of colors in the color map (default = 256)
%
% OUTPUTS:
%       cm          - three column color map matrix which can be applied
%                     using colormap
%
% ABOUT:
%       author      - Bradley Treeby
%       date        - 3rd July 2009
%       last update - 17th July 2009
%
% MODIFICATIONS:
%       author      - Manuel Diaz
%       date        - 28th June 2016
%       last update - 1th July 2016
%       
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2014 Bradley Treeby and Ben Cox

%
if nargin < 1
   cmap = 'default'; % Bradley Treeby, K-wave
elseif nargin >= 1
   cmap = varargin{1}; % JetVar   
end

% 
switch cmap
    case 'kwave'
        
        % Set literals
        neg_pad = 48;
        num_colors = 256; % default numer of colors!
        
        % Define colour spectrums
        neg = bone(num_colors/2 + neg_pad);
        neg = neg(1 + neg_pad:end, :);
        pos = flipud(hot(num_colors/2));
        
        % Create custom colour map
        cm = [neg; pos];

    case 'positivePressure'

        % Create custom colour map
        cm = flipud(hot(256));
        
    case 'negativePressure'
               
        % Create custom colour map
        cm = bone(256);
        
    case 'cold'
        
        m = size(get(gcf,'colormap'),1);
        n = fix(3/8*m);
        
        r = [zeros(2*n,1); (1:m-2*n)'/(m-2*n)];
        g = [zeros(n,1); (1:n)'/n; ones(m-2*n,1)];
        b = [(1:n)'/n; ones(m-n,1)];
        
        cm= [r,g,b];
        
    otherwise % set Jetvar!
        
        m = size(get(gcf, 'colormap'), 1);
        out = jet(m);
        
        % Modify the output starting at 1 before where Jet outputs pure blue.
        n = find(out(:,3) == 1,1)-1;
        out(1:n,1:2) = repmat((n:-1:1)'/n,[1 2]);
        out(1:n,3) = 1;
        
        cm = out;
        
end