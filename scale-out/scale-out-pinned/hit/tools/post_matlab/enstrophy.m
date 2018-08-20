function enstro = enstrophy(u,v,w)
  ox = fou2phys(ddy(w) - ddz(v));
  oy = fou2phys(ddz(u) - ddx(w));
  oz = fou2phys(ddx(v) - ddy(u));

  enstro = ox.^2 + oy.^2 + oz.^2;
