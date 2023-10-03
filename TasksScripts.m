% %   OPTIMIZATION AND ALGORITHMS 2023/2024
%  
% %   INSTITUTO SUPERIOR TÉCNICO MEEC
% 
% %   Author: João Castelo Branco, Mike, Rodas e Bernas
% 
% %   This file contains the Matlab resolution of Tasks 1-16 of the course's
% %   project.

%% Task1 
clear;
close all;
clc;

% Define the anchor location and measured range
a = 2;
r = 1;

% Define a range of x values for plotting
x = linspace(-2, 6, 500);  

% Calculate the cost function: (|x - a| - r)^2
cost = (abs(x - a) - r).^2;

% Plot the cost function
plot(x, cost, 'LineWidth',2,'DisplayName', 'Cost Function');
hold;
plot(2, 0, '.', 'MarkerSize', 15, 'DisplayName', 'Anchor'); % Display an Anchor
set(gca, 'YLim', [-0.25, 2]); % Set Limits on Y Axis
xlabel('x');
ylabel('Cost');
title('Cost Function: (|x - a| - r)^2');
grid on;
legend('Location', 'Best'); % Add Legend
saveas(gcf,"Task1.png");


% Comment: The cost function ( |x - a| - r )^2 indicates inconsistency
% between predicted and measured ranges for a target's position estimation.
% With symmetry around a = 2, it exhibits multiple minima (x=1 and x=3), implying ambiguity
% in localizing the target. The cost sharply increases away from these minima,
% emphasizing sensitivity to position deviations. The target may be in one
% of those positions.



%% Task2
clear;
close all;
clc;

% Define anchor locations and range measurements
a1 = [-1, 0];
a2 = [3, 0];
r1 = 2;
r2 = 3;

% Define a grid of x and y values for contour plot
x = -3:0.1:5;
y = -3:0.1:3;
[X, Y] = meshgrid(x, y);

% Initialize the cost function matrix
cost = zeros(size(X));

% Calculate the cost function for each point in the grid
for i = 1:numel(X)
    x_current = [X(i), Y(i)]; % Current (x, y) position
    cost(i) = ((norm(x_current - a1) - r1)^2) + ((norm(x_current - a2) - r2)^2);
end

% Create a contour plot of the cost function
figure;
c = contour(X, Y, cost, 300); % Adjust the number of contour lines as needed
hold;
plot(a1(1), a1(2), 'r.', 'MarkerSize', 20); % Display an Anchor
plot(a2(1), a2(2), 'r.', 'MarkerSize', 20); % Display an Anchor
xlabel('x');
ylabel('y');
title('Cost Function Contour Plot');
colorbar;
saveas(gcf,"Task2.png");

% Create a 3D plot of the cost function
figure;
surf(X, Y, cost);
xlabel('x');
ylabel('y');
zlabel('Cost');
title('Cost Function 3D Plot');


% Minimizer Uniqueness: 
% The minimizer for this problem is not unique because there are two intersection 
% points in the 2D plane, resulting in two potential target positions. 
% This non-uniqueness arises from the inherent geometric nature of intersecting circles.
% 
% Choosing Anchors for a Unique Minimizer: 
% To achieve a cost function with a 
% single global minimizer (excluding "degenerate" configurations), you would 
% need a minimum of three non-collinear anchor points. With three such anchors 
% and their respective range measurements, you can uniquely determine the 
% target's position using trilateration, as it involves solving a system of 
% equations with three unknowns (x, y, and potentially z in 3D) and three 
% equations derived from the range measurements. This provides a unique 
% solution and eliminates the ambiguity present in the two-anchor scenario.




%% Task3a - Repetition of task 1 with convex approximation
clear;
close all;
clc;

% Define the anchor location and measured range
a = 2;
r = 1;

% Define a range of x values for plotting
x = linspace(0, 4, 500);

% Calculate the cost function using the convex approximation
cost = max(abs(x - a) - r, 0).^2;

% Plot the cost function
plot(x, cost, 'LineWidth',2,'DisplayName', 'Cost Function');
hold;
plot(a, 0, '.', 'MarkerSize', 15, 'DisplayName', 'Anchor'); % Display an Anchor
set(gca, 'YLim', [-0.25, 1.4]); % Set Limits on Y Axis
xlabel('x');
ylabel('Cost');
title('Cost Function: max(|x - a| - r, 0)^2');
grid on;
legend('Location', 'Best'); % Add Legend
saveas(gcf,"Task3a.png");



%% Task3b - Repetition of task 2 with convex approximation
clear;
close all;
clc;

% Define anchor locations and range measurements
a1 = [-1, 0];
a2 = [3, 0];
r1 = 2;
r2 = 3;

% Define a grid of x and y values for contour plot
x = -3:0.1:6;
y = -3:0.1:3;
[X, Y] = meshgrid(x, y);

% Initialize the cost function matrix
cost = zeros(size(X));

% Calculate the cost function using the convex approximation
for i = 1:numel(X)
    x_current = [X(i), Y(i)]; % Current (x, y) position
    cost(i) = max(norm(x_current - a1) - r1, 0)^2 + max(norm(x_current - a2) - r2, 0)^2;
end

% Create a contour plot of the cost function
figure;
contour(X, Y, cost, 100); % Adjust the number of contour lines as needed
hold;
plot(a1(1), a1(2), 'r.', 'MarkerSize', 20); % Display an Anchor
plot(a2(1), a2(2), 'r.', 'MarkerSize', 20); % Display an Anchor
xlabel('x');
ylabel('y');
title('Cost Function Contour Plot (Convex Approximation)');
colorbar;
saveas(gcf,"Task3b.png");

%Cost function
figure;
surf(X, Y, cost);
xlabel('x');
ylabel('y');
zlabel('Cost');
title('Cost Function 3D Plot');


% In both tasks 3a and 3b, applying convex approximations to the cost functions 
% did not reduce the ambiguity in solutions; it may have even increased it. 
% The inherent problem of multiple potential solutions in localization scenarios
% with intersecting circles persisted, highlighting that convex relaxations 
% may not eliminate ambiguity.



%% Task 4a 
clear;
close all;
clc;

%cvx_quiet(true); % Suppress CVX output

% Define the anchor location and measured range
a = 2;
r = 1;

% Create CVX variable
cvx_begin quiet
    variable x(1)
    
    % Define the cost function using the convex approximation
    minimize (square_pos(norm(x - a) - r))
cvx_end

% Display the results
disp('CVX Solution:');
disp(['Target Position: x = ' num2str(x) ' ']);




%% Task 4b - Solve the 2D localization problem in CVX with convex approximation
clear;
close all;
clc;

% Define anchor locations and range measurements
a1 = [-1, 0];
a2 = [3, 0];
r1 = 2;
r2 = 3;

% Create CVX variables for the target's 2D position
cvx_begin quiet
    variable x(2)
    
    % Define the cost function using the convex approximation
    minimize (square_pos(norm(x - a1') - r1) + square_pos(norm(x - a2') - r2))
cvx_end

% Display the results
disp('CVX Solution:');
disp(['Target Position (x, y): (' num2str(x(1)) ', ' num2str(x(2)) ')']);



% In the context of localization problems, it's important to note that applying 
% the convex approximation, as demonstrated in Task 4, indeed imparts convexity 
% to the optimization problems. This feature eliminates the need to provide an 
% initial estimate of the solution, as CVX can effectively handle convex 
% optimization tasks.
% 
% However, a critical observation emerges: the use of convex approximation, 
% while ensuring convexity, may increase the inherent ambiguity of the solution 
%     in localization problems. As a result, the optimization process may yield 
%     ambiguous or incorrect results, as seen in Task 4a and 4b.
% 
% In summary, the convex approximation, although suitable for convex 
% optimization, may not fully resolve the fundamental issue of ambiguity in 
% localization problems. Localization problems, by their nature, can still 
% exhibit multiple potential solutions, leading to incorrect outcomes even 
% when convex optimization tools like CVX are employed. Therefore, for 
% localization tasks with inherent ambiguity, it remains essential to consider 
% specialized non-convex optimization techniques and provide initial estimates 
% to guide the optimization process towards accurate solutions.


%% Task 5

% Exercício teórico: resolver em papel? Como introduzir no relatório?
% 
% Nota: Foi resolvido nas aulas teóricas e a solução está antes da task 6...



%% Task 6

% Exercício teórico

% Incorporating angular information into localization problems offers a 
% powerful means to disambiguate solutions. While relaxing the range terms in
% the cost function provides convexity, it can increase the ambiguity of the 
% solution. However, by adding angular information through distinct directions 
% (uk), we can significantly enhance the problem's clarity.
% 
% If all directions (uk) are distinct, the minimum number required to ensure 
% a unique solution, regardless of the range terms, is two. In a 2D localization 
% problem, having two unique directions allows us to determine both the 
% position and orientation of the target uniquely. This is achieved by 
% removing angular ambiguity, as the two directions provide orthogonal 
% reference frames.
% 
% Importantly, to guarantee a unique solution, it is essential that the 
% angular direction of the target relative to each of these two possible 
% anchors is not the same. In other words, the three vectors (directions 
% from the target to the two anchors and their relative positions) should 
% not be collinear. This condition ensures that the angular information 
% provided by the distinct directions is non-redundant and contributes to 
% the unequivocal determination of the target's position and orientation.


%% Task 7a
% Scenario (i) - Single Anchor
clear;
close all;
clc;

% Define anchor location and angle in polar coordinates (40 degrees)
a = [-1, 0];
u_angle = deg2rad(40); % Angle in radians

% Create a grid of x and y values for the contour plot
x = -3:0.1:2;
y = -2:0.1:2;
[X, Y] = meshgrid(x, y);

% Initialize the cost function matrix
cost = zeros(size(X));

% Calculate the cost function for each point in the grid
for i = 1:numel(X)
    x_current = [X(i), Y(i)]; % Current (x, y) position
    direction = [cos(u_angle), sin(u_angle)]; % Convert polar angle to direction vector
    cost(i) = norm((eye(2) - direction' * direction) * (x_current - a)')^2;
end

% Create a contour plot of the cost function
figure;
contour(X, Y, cost, 100); % Adjust the number of contour lines as needed
xlabel('x');
ylabel('y');
title('Angular Cost Function Contour Plot (Single Anchor)');
colorbar;

% Add marker for anchor position
hold on;
plot(a(1), a(2), 'r.', 'MarkerSize', 15);
hold off;

saveas(gcf,"Task7a.png");


%% 7b

% Scenario (ii) - Two Anchors
clear;
close all;
clc;

% Define anchor locations and angles in polar coordinates (40 degrees and 140 degrees)
a1 = [-1, 0];
a2 = [3, 0];
u1_angle = deg2rad(40); % Angle in radians
u2_angle = deg2rad(140); % Angle in radians

% Create a grid of x and y values for the contour plot
x = -6:0.1:9;
y = -2:0.1:5;
[X, Y] = meshgrid(x, y);

% Initialize the cost function matrix
cost = zeros(size(X));

% Calculate the cost function for each point in the grid
for i = 1:numel(X)
    x_current = [X(i), Y(i)]; % Current (x, y) position
    direction1 = [cos(u1_angle), sin(u1_angle)]; % Convert polar angle to direction vector for u1
    direction2 = [cos(u2_angle), sin(u2_angle)]; % Convert polar angle to direction vector for u2
    cost1 = norm((eye(2) - direction1' * direction1) * (x_current - a1)')^2;
    cost2 = norm((eye(2) - direction2' * direction2) * (x_current - a2)')^2;
    cost(i) = cost1 + cost2;
end

% Create a contour plot of the cost function
figure;
contour(X, Y, cost, 100); % Adjust the number of contour lines as needed
xlabel('x');
ylabel('y');
title('Angular Cost Function Contour Plot (Two Anchors)');
colorbar;

% Add markers for anchor positions
hold on;
plot(a1(1), a1(2), 'r.', 'MarkerSize', 15);
plot(a2(1), a2(2), 'r.', 'MarkerSize', 15);
hold off;

saveas(gcf,"Task7b.png");



%% Task 8

clear;
close all;
clc;

% Define anchor locations and range measurements
a1 = [-1, 0];
a2 = [3, 0];
r1 = 2;
r2 = 3;

% Create CVX variables for the target's 2D position
cvx_begin quiet
    variable x(2)
    
    % Define the cost function using the convex approximation with angular terms
    term1 = square_pos(norm(x - a1') - r1);
    term2 = square_pos(norm(x - a2') - r2);
    angular_term1 = norm((eye(2) - [cosd(40); sind(40)]*[cosd(40) sind(40)])*(x - a1'));
    angular_term2 = norm((eye(2) - [cosd(140); sind(140)]*[cosd(140) sind(140)])*(x - a2'));
    
    minimize (term1 + term2 + angular_term1 + angular_term2)
cvx_end

% Display the results
disp('CVX Solution:');
disp(['Target Position (x, y): (' num2str(x(1)) ', ' num2str(x(2)) ')']);


% In comparing the outcomes of Task 4b and its angular version, it becomes 
% evident that the inclusion of angular measurements significantly reduces 
% ambiguity in the localization problem. In the initial version, without angular 
% information, the solution exhibited notable ambiguity, yielding 
% 'Target Position (x, y): (0.50991, 0).' This result indicated the challenges 
% posed by the inherent ambiguity in range-based localization.
% 
% However, in the angular version with angular measurements at 40° and 140°, 
% the ambiguity is notably diminished. The solution obtained was 
% 'Target Position (x, y): (0.90929, 1.6021),' which aligns more closely 
% with our expectations and reflects a more reasonable estimate of the 
% target's position. This underscores the value of incorporating angular data 
% into localization problems to enhance the accuracy and reliability of the 
% solutions obtained



%% Task 9 

close all
clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the bounding box and scaling factors

% x_scale = 0.18107;
% y_scale = 0.21394;
% xx_org = 235;
% yy_org = 258;

figure;
xlim([-21, 21]);
ylim([-21, 21]);
grid on;
hold on;
xlabel('X (meters)');
ylabel('Y (meters)');
title('Generated 2D Trajectory with Anchors');


x_min = -20;
x_max = 20;
y_min = -20;
y_max = 20;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate the bounding box vertices
box_vertices = [x_min, y_min; x_min, y_max; x_max, y_min; x_max, y_max];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Place static anchors near the box vertices (adjust positions as needed)
anchors = box_vertices - 1 + 2*rand(4, 2);
plot(anchors(:, 1), anchors(:, 2), 'bo', 'MarkerSize', 7, 'MarkerFaceColor', 'b');

% Plot the actual bounding box
rectangle('Position', [x_min, y_min, x_max - x_min, y_max - y_min], 'LineWidth', 0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining trajectory mandatory points

disp(' ')
disp('Use the mouse to input via points for the reference trajectory.');
disp('Press --button 3-- to end the input.');
button = 1;
k = 1;

while button == 1
    [x(k), y(k), button] = ginput(1);
    
    % Scale and translate coordinates to match the bounding box
    % x(k) = (x(k) - xx_org) * x_scale;
    % y(k) = (y(k) - yy_org) * y_scale;
    
    % Ensure points stay within the bounding box
    x(k) = max(min(x(k), x_max), x_min);
    y(k) = max(min(y(k), y_max), y_min);
    
    plot(x(k), y(k), 'r+', 'Linewidth', 2);
    k = k + 1;
end

drawnow;
disp(' ')
disp(['There are ', num2str(k-1), ' points to interpolate from.'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Trajectory generation using cubic splines

h = 0.01;
npt = length(x);        % number of via points, including initial and final
nvia = 0:1:npt-1;

csinterp_x = csapi(nvia, x);
csinterp_y = csapi(nvia, y);

time = 0:h:npt-1;
xx = fnval(csinterp_x, time);
yy = fnval(csinterp_y, time);

% Scale and translate the generated trajectory back to world coordinates
% xx = xx / x_scale + xx_org;
% yy = yy / y_scale + yy_org;

% Plot the generated trajectory 
plot(xx, yy);

% Simulate target motion and record measurements
T = 100; % Total number of samples
sample_rate = 2; % Hz
dt = 1 / sample_rate;
speed = 1.5; % m/s 
trajectory = [xx; yy];

% Initialize arrays to store measurements
ranges = zeros(T, 4); % Four anchors
angles = zeros(T, 4); % Four anchors
velocities = zeros(T, 1);

% ESTA PARTE AINDA NÃO ESTÁ BEM MAS DEVO ACABAR EM BREVE

for t = 1:T
    % Simulate target motion
    if t > 1
        % Compute velocity from consecutive positions
        delta_position = trajectory(:, t) - trajectory(:, t - 1);
        velocity = norm(delta_position) / dt;
    else
        velocity = speed; %initial speed
    end
    
    velocities(t) = velocity;
    
    % Compute range and angle measurements for each anchor
    for anchor_idx = 1:4
        anchor_position = anchors(anchor_idx, :);
        target_position = trajectory(:, t)';
        delta = target_position - anchor_position;
        range = norm(delta);
        angle = atan2(delta(2), delta(1));
        
        ranges(t, anchor_idx) = range;
        angles(t, anchor_idx) = rad2deg(angle); %save angle in degrees
    end
end

















