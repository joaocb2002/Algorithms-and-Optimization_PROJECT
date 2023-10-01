%
% trajectory generation test using cubic splines and
% plot of a rectangular vehicle over the trajectory
% 
% Robotics 2021, Joao S. Sequeira
%

close all
clear
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the reference trajectory;
figure(1)
hold on
grid on

x_scale = 0.18107;
disp(['xx scale factor ', num2str(x_scale), ' meters/pixel']);

y_scale = 0.21394;
disp(['yy scale factor ', num2str(y_scale), ' meters/pixel']);

xx_org = 235;
yy_org = 258;
disp(['World frame origin at image coordinates ', num2str(xx_org), ' ', num2str(yy_org)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% starting of the trajectory generation reference stuff
%
% each coordinate is interpolated, using cubic polynomials, from a set of
% via points, plus initial and final point

disp(' ')
disp('Use the mouse to input via points for the reference trajectory.');
disp('Press --button 3-- to end the input.');
button = 1;
k = 1;

while button==1
    [x(k),y(k), button] = ginput(1);
    plot(x(k),y(k),'r+', 'Linewidth', 2);
    k = k + 1;
end

drawnow;
disp(' ')
disp(['There are ', num2str(k-1), ' points to interpolate from.'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select the anchors using the mouse

% disp(' ')
% disp('Use the mouse to input via points for the reference trajectory.');
% disp('Press --button 3-- to end the input.');
% button = 1;
% k = 1;
% while button==1,
%     [ax(k),ay(k),button] = ginput(1);
%     plot(ax(k),ay(k),'bo')
%     anchor(k,:) = [ax(k), ay(k)];
%     k = k + 1;
% end
% drawnow;
% disp([ num2str(k-1), ' anchors'])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the coefficients of the cubic polynomial are computed in the csapi
% function
% the evaluation of the polynomial for suitable instants of time is done by
% the neval functions

h = 0.01;
npt = length(x);        % number of via points, including initial and final
nvia = [0:1:npt-1];
csinterp_x = csapi(nvia,x);
csinterp_y = csapi(nvia,y);
time = [0:h:npt-1];
xx = fnval(csinterp_x, time);
yy = fnval(csinterp_y, time);
plot(xx,yy)


