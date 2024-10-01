function dx = pendcart(x,m,M,L,g,d,u)
Sx = sin(x(3));
Cx = cos(x(3));
D = m*L*L*(M+m*(1-Cxˆ2));
dx(1,1) = x(2);
dx(2,1) = (1/D)*(-mˆ2*Lˆ2*g*Cx*Sx + m*Lˆ2*(m*L*x(4)^2*Sx - d*x(2))) + m*L*L*(1/D)*u;
dx(3,1) = x(4);
dx(4,1) = (1/D)*((m+M)*m*g*L*Sx - m*L*Cx*(m*L*x(4)^2*Sx - d*x(2))) - m*L*Cx*(1/D)*u;
