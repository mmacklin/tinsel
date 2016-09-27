Le = 1.0
f = 1.0/pi
r = 1.0
d = 10.0
alpha = asin(r/d)

s = @(phi, theta) f.*Le.*cos(theta).*sin(theta)

Lo = integral2(s, 0, 2*pi, 0, alpha)