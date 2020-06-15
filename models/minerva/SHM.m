
timestep = 0.01;
runtime = 10;

m = 1; % mass in kg
k = 1; % spring constant in N/m
xo = 1; % initial position in m

a = zeros(1,runtime/timestep);
v = zeros(1,runtime/timestep);
x = zeros(1,runtime/timestep);
pe = zeros(1,runtime/timestep);
ke = zeros(1,runtime/timestep);
time = zeros(1,runtime/timestep);

a(1,1) = -(k * xo) / m;
v(1,1) = 0;
x(1,1) = xo;
ke(1,1) = 0;
pe(1,1) = 0.5 * k * xo^2;

for i = 2 : runtime/timestep
    time(1,i) = time(1,i-1) + timestep;
    a(1,i) = -(k * x(1,i-1)) / m;
    v(1,i) = v(1,i-1) + a(1,i-1) * timestep;
    x(1,i) = x(1,i-1) + v(1,i-1) * timestep;
    ke(1,i) = 0.5 * m * v(1,i)^2;
    pe(1,i) = 0.5 * k * x(1,i)^2;
end

plot(time,a)
hold on;
plot(time,v)
plot(time,x)
hold off;
legend('acceleration','velocity','position')