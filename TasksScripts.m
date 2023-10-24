% %   OPTIMIZATION AND ALGORITHMS 2023/2024
%  
% %   INSTITUTO SUPERIOR TÉCNICO MEEC
% 
% %   Author: João Castelo Branco, Miguel Lameiras, Rodrigo Faria e Bernardo Soares
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
c = contour(X, Y, cost, 50); % Adjust the number of contour lines as needed
%surf(X,Y,cost)
hold;
%plot(a1(1), a1(2), 'r.', 'MarkerSize', 20); % Display an Anchor
%plot(a2(1), a2(2), 'r.', 'MarkerSize', 20); % Display an Anchor
xlabel('x');
ylabel('y');
title('Cost Function Contour Plot');
colorbar;
saveas(gcf,"Task2.png");

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

x = linspace(0, 4, 500);
cost = max(abs(x - a) - r, 0).^2;

% Create CVX variable
cvx_begin quiet
    variable xs(1)
    
    % Define the cost function using the convex approximation
    minimize (square_pos(norm(xs - a) - r))
cvx_end

% Display the results
disp('CVX Solution:');
disp(['Target Position: x = ' num2str(xs) ' ']);

plot(x, cost, 'LineWidth',2,'DisplayName', 'Cost Function');
hold;
plot(xs, 0, '.', 'MarkerSize', 15, 'DisplayName', 'CVX Solution'); % Display an Anchor
set(gca, 'YLim', [-0.25, 2]); % Set Limits on Y Axis
xlabel('x');
ylabel('Cost');
title('Cost Function: (|x - a| - r)^2 Convex Aproximation');
grid on;
legend('Location', 'Best'); % Add Legend
saveas(gcf,"Task4a.png");




%% Task 4b - Solve the 2D localization problem in CVX with convex approximation
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

% Create CVX variables for the target's 2D position
cvx_begin quiet
    variable xs(2)
    
    % Define the cost function using the convex approximation
    minimize (square_pos(norm(xs - a1') - r1) + square_pos(norm(xs - a2') - r2))
cvx_end

% Display the results
disp('CVX Solution:');
disp(['Target Position (x, y): (' num2str(x(1)) ', ' num2str(x(2)) ')']);

% Create a contour plot of the cost function
figure;
contour(X, Y, cost, 100); % Adjust the number of contour lines as needed
hold;
plot(xs(1), xs(2), 'r.', 'MarkerSize', 20); % Display an Anchor
xlabel('x');
ylabel('y');
title('Cost Function Contour Plot (Convex Approximation)');
colorbar;
saveas(gcf,"Task4b.png");


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

direction = [cos(u_angle), sin(u_angle)]; % Convert polar angle to direction vector

% Calculate the cost function for each point in the grid
for i = 1:numel(X)
    x_current = [X(i), Y(i)]; % Current (x, y) position
    cost(i) = norm((eye(2) - direction' * direction) * (x_current - a)')^2;
end

% Create a contour plot of the cost function
figure;
contour(X, Y, cost, 100); % Adjust the number of contour lines as needed
%surf(X,Y,cost)
xlabel('x');
ylabel('y');
title('Angular Cost Function Contour Plot - u = 40º');
colorbar;

% Add marker for anchor position
hold on;
quiver(a(1), a(2), direction(1), direction(2), 'b', 'LineWidth', 1, 'MaxHeadSize', 0.5);
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

direction1 = [cos(u1_angle), sin(u1_angle)]; % Convert polar angle to direction vector for u1
direction2 = [cos(u2_angle), sin(u2_angle)]; % Convert polar angle to direction vector for u2

% Calculate the cost function for each point in the grid
for i = 1:numel(X)
    x_current = [X(i), Y(i)]; % Current (x, y) position
    cost1 = norm((eye(2) - direction1' * direction1) * (x_current - a1)')^2;
    cost2 = norm((eye(2) - direction2' * direction2) * (x_current - a2)')^2;
    cost(i) = cost1 + cost2;
end

% Create a contour plot of the cost function
figure;
contour(X, Y, cost, 100); % Adjust the number of contour lines as needed
%surf(X,Y,cost);
xlabel('x');
ylabel('y');
title('Angular Cost Function Contour Plot - u1 = 40º and u2 = 140º');
set(gca, 'YLim', [-2, 5]); % Set Limits on Y Axis
colorbar;

% Add markers for anchor positions
hold on;
quiver(a1(1), a1(2), direction1(1), direction1(2), 'b', 'LineWidth', 1, 'MaxHeadSize', 0.5);
quiver(a2(1), a2(2), direction2(1), direction2(2), 'b', 'LineWidth', 1, 'MaxHeadSize', 0.5);
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



%% Task 9 and 10 - Trajectory Estimation with Motion

close all;
clear;
clc;

figure;
xlim([-21, 21]);
ylim([-21, 21]);
grid on;
hold on;
xlabel('X (meters)');
ylabel('Y (meters)');
tl1 = title('Generated 2D Trajectory with Anchors');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the bounding box and scaling factors
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

colors = ["r","g","b","m"];
for i = 1:4
    plot(anchors(i, 1), anchors(i, 2), 'bo', 'MarkerSize', 7, 'MarkerFaceColor', colors(i));
end
% Plot the actual bounding box
rectangle('Position', [x_min, y_min, x_max - x_min, y_max - y_min], 'LineWidth', 0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Defining trajectory mandatory points

disp(' ')
disp('Use the mouse to input via points for the reference trajectory.');
disp('Press Space key to end the input.');
disp(' ');
k = 1;


while 1
    [x(k), y(k), button] = ginput(1);
    
    if button == 32 % 32 = Space key
        x(k) = [];  % delete last point acquired when space is pressed
        y(k) = [];  % delete last point acquired when space is pressed
        break
    end

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

npt = length(x);        % number of via points, including initial and final
nvia = 0:1:npt-1;

csinterp_x = csapi(nvia, x);
csinterp_y = csapi(nvia, y);

T = 100; %number of samples
time = linspace(0, npt-1, T); 
xx = fnval(csinterp_x, time);
yy = fnval(csinterp_y, time);

% Plot the generated trajectory ('real one')
plot(xx, yy, 'ro-');
saveas(gcf,"Task9.png");
%disp("here")
% Simulate target motion and record measurements
sample_rate = 2; % Hz
dt = 1 / sample_rate;
trajectory = [xx; yy];

% Initialize arrays to store measurements
ranges = zeros(T, 4); % Four anchors
angles = zeros(T, 4); % Four anchors
velocities = zeros(2, T);

for t = 1:T
    % Simulate target motion
    if t == 1
        delta_position = trajectory(:, t+1) - trajectory(:, t); %initial speed
        velocity = (delta_position) / dt; 
    elseif t == T 
        delta_position = trajectory(:, t-1) - trajectory(:, t); %final speed
        velocity = (delta_position) / dt;  
    else
        % Compute velocity from consecutive positions
        delta_position = trajectory(:, t+1) - trajectory(:, t-1); %formula 3 do enunciado
        velocity = (delta_position) / (2*dt);
    end
    
    velocities(:, t) = 0.8*velocity; %Set of velocity vectors (0.8 introduces an inconsistency)
    
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

% We saved ranges, angles and velocities

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define optimization variables and find optimal trajectory for obtained
% measurements

mu_values = [0.01,0.1,1,10,100,1000];

for i = 1:6
    disp("Computing Solution fo mu = " + string(mu_values(i)) + "...");
    mu = mu_values(i);
    cvx_begin quiet
        variable x(2, T)
        
        % Define the cost function
        cost = 0;
        for t = 1:T
            
            %Range measurements part
            for anchor_idx = 1:4
                anchor_position = anchors(anchor_idx, :);
                delta = x(:, t) - anchor_position';
                range = norm(delta);
                cost = cost + square_pos(range - ranges(t, anchor_idx));
            end
            
            %Velocities part
            if t == T
                veloc = (x(:, t-1) - x(:, t))/dt;
            elseif t == 1
                veloc = (x(:, t+1) - x(:, t))/dt;
            else
                veloc = (x(:, t+1) - x(:, t-1))/(2*dt);
            end
                
            cost = cost + mu * square_pos(norm(veloc - velocities(:, t)));
        end
        
        minimize(cost)  
    cvx_end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plot the optimizated trajectory 
    pl = plot(x(1,:),x(2,:),"bx");
    delete(tl1);
    tl = title('Generated 2D Trajectory with Anchors with \mu = ' + string(mu));
    saveas(gcf,"Task10_mu=" + string(mu) + ".png")
    delete(pl);
    delete(tl);
end

hold off;

figure;
hold on;
plot(time,velocities(1,:),"-r","LineWidth",1);
plot(time,velocities(2,:),"-b","LineWidth",1);
plot(time,sqrt(sum(velocities.^2)),"-m","LineWidth",1);
title("Velocities");
xlabel("Time [s]");
ylabel("Velocity [m/s]");
hold off;
legend("x-axis velocity","y-axis velocity","velocity norm");
grid on;
saveas(gcf,'Velocities.png')

figure;
hold on;
plot(time,ranges(:,1),"-r","LineWidth",1);
plot(time,ranges(:,2),"-g","LineWidth",1);
plot(time,ranges(:,3),"-b","LineWidth",1);
plot(time,ranges(:,4),"-m","LineWidth",1);
title("Range Measurements");
xlabel("Time [s]");
ylabel("Distance to Anchor [m]");
hold off;
grid on;
saveas(gcf,'Range_Measurements.png')

figure;
hold on;
plot(time,angles(:,1),"-r","LineWidth",1);
plot(time,angles(:,2),"-g","LineWidth",1);
plot(time,angles(:,3),"-b","LineWidth",1);
plot(time,angles(:,4),"-m","LineWidth",1);
title("Angle Measurements");
xlabel("Time [s]");
ylabel("Angle Measurement [º]");
hold off;
grid on;
saveas(gcf,'Angle_Measurements.png')

% Task 10 conclusion
%
% The objective of Task 10 is to investigate the influence of the parameter μ 
% on the trajectory estimation problem. By varying the value of μ, we can observe 
% how the optimization process balances the significance of the velocity 
% measurements against the range measurements. This allows us to evaluate 
% the trade-off between these two sources of information in the trajectory 
% estimation process.
% 
% When we set μ to a very small value, such as μ = 0.01, the optimization 
% trajectory closely aligns with the trajectory defined by interpolation. 
% In this scenario, the term in the cost function associated with velocity 
% measurements has a relatively low impact. Essentially, it suggests that 
% adhering to velocity constraints is not a critical aspect of the optimization. 
% Consequently, the solution predominantly adheres to the range measurements 
% in the cost function.
% 
% Conversely, when μ is set to a significantly larger value, such as μ = 100, 
% the optimization trajectory appears to be smaller in scale compared to the 
% interpolation-based trajectory. This effect arises from the heightened 
% importance of the term associated with velocity measurements in the cost 
% function. The requirement to satisfy velocity constraints takes precedence 
% over the range constraints in this case.
% 
% In summary, the parameter μ plays a pivotal role in determining the relative 
% significance of velocity measurements in the cost function. A larger μ value 
% places greater importance on velocity requirements, potentially leading to 
% solutions that exhibit inconsistencies with the trajectory derived from range 
% measurements. The choice of μ thus provides a means to fine-tune the balance 
% between the impact of range and velocity measurements in the trajectory 
% estimation process.



%% Task 11
close all;
clear;
clc;

figure(1);
xlim([-21, 21]);
ylim([-21, 21]);
grid on;
hold on;
xlabel('X (meters)');
ylabel('Y (meters)');
title('Generated 2D Trajectory with Anchors');

%Bounding box
x_min = -20;
x_max = 20;
y_min = -20;
y_max = 20;
box_vertices = [x_min, y_min; x_min, y_max; x_max, y_min; x_max, y_max];

%Anchors
anchors = box_vertices - 1 + 2*rand(4, 2);
plot(anchors(:, 1), anchors(:, 2), 'bo', 'MarkerSize', 7, 'MarkerFaceColor', 'b');

% Plot the actual bounding box and defining trajectory mandatory points
rectangle('Position', [x_min, y_min, x_max - x_min, y_max - y_min], 'LineWidth', 0.5);
disp(' ')
disp('Use the mouse to input via points for the reference trajectory.');
disp('Press Space key to end the input.');
disp(' ');
k = 1;

while 1
    [x(k), y(k), button] = ginput(1);
    
    if button == 32 % 32 = Space key
        x(k) = [];  % delete last point acquired when space is pressed
        y(k) = [];  % delete last point acquired when space is pressed
        break
    end

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

npt = length(x);        % number of via points, including initial and final
nvia = 0:1:npt-1;

csinterp_x = csapi(nvia, x);
csinterp_y = csapi(nvia, y);

T = 75; %number of samples
time = linspace(0, npt-1, T); 
xx = fnval(csinterp_x, time);
yy = fnval(csinterp_y, time);

% Plot the generated trajectory ('real one')
plot(xx, yy, 'ro-');

% Simulate target motion and record measurements
sample_rate = 2; % Hz
dt = 1 / sample_rate;
trajectory = [xx; yy];

% Initialize arrays to store measurements
ranges = zeros(T, 4); % Four anchors
angles = zeros(T, 4); % Four anchors
velocities = zeros(2, T);

for t = 1:T
    % Simulate target motion
    if t == 1
        delta_position = trajectory(:, t+1) - trajectory(:, t); %initial speed
        velocity = (delta_position) / dt; 
    elseif t == T 
        delta_position = trajectory(:, t-1) - trajectory(:, t); %final speed
        velocity = (delta_position) / dt;  
    else
        % Compute velocity from consecutive positions
        delta_position = trajectory(:, t+1) - trajectory(:, t-1); %formula 3 do enunciado
        velocity = (delta_position) / (2*dt);
    end
    
    velocities(:, t) = 0.8*velocity; %Set of velocity vectors (0.8 introduces an inconsistency)
    
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

% We saved ranges, angles and velocities from the 'real trajectory'



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define optimization variables and find optimal trajectory for obtained
% measurements and determine range and velocity error calculation

values = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000]; % Value of mu

RE = zeros(1, length(values));
VE = zeros(1, length(values));

for k = 1:length(values)
    mu = values(k);
    
   
    cvx_begin quiet
        variable x(2, T)

        % Define the cost function
        cost = 0;
        for t = 1:T

            %Range measurements part
            for anchor_idx = 1:4
                anchor_position = anchors(anchor_idx, :);
                delta = x(:, t) - anchor_position';
                range = norm(delta);
                cost = cost + square_pos(range - ranges(t, anchor_idx));
            end

            %Velocities part
            if t == T
                veloc = (x(:, t-1) - x(:, t))/dt;
            elseif t == 1
                veloc = (x(:, t+1) - x(:, t))/dt;
            else
                veloc = (x(:, t+1) - x(:, t-1))/(2*dt);
            end

            cost = cost + mu * square_pos(norm(veloc - velocities(:, t)));
        end

        minimize(cost)  
    cvx_end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plot the optimizated trajectory 
    figure(k+1);
    xlim([-21, 21]);
    ylim([-21, 21]);
    grid on;
    hold on;
    xlabel('X (meters)');
    ylabel('Y (meters)');
    title(['Optimization 2D Trajectory for \mu = ', num2str(mu)]);
    plot(anchors(:, 1), anchors(:, 2), 'bo', 'MarkerSize', 7, 'MarkerFaceColor', 'b');
    rectangle('Position', [x_min, y_min, x_max - x_min, y_max - y_min], 'LineWidth', 0.5);
    plot(xx, yy, 'ro-'); %Real Trajectory
    
    for t = 1:T
        plot(x(1, t), x(2, t), 'bx'); %Optimization trajectory
    end

    for i=1:T 
        for anchor_idx = 1:4
            anchor_position = anchors(anchor_idx, :);
            delta = x(:, t) - anchor_position';
            range = norm(delta);
            RE(1, k) = RE(1, k) + (abs((range - ranges(t, anchor_idx))))^2;
        end

        if t == T
            veloc = (x(:, t-1) - x(:, t))/dt;
        elseif t == 1
            veloc = (x(:, t+1) - x(:, t))/dt;
        else
            veloc = (x(:, t+1) - x(:, t-1))/(2*dt);
        end

        VE(1, k) = (norm(veloc - velocities(:, t)))^2;

    end
end

% Plot the Range Error (RE) vs Velocity Error (VE)
figure;
semilogx(RE, VE, 'ro-');
title('2D plot for RE(\mu) vs VE(\mu)');
xlabel('Range Error (RE)');
ylabel('Velocity Error (VE)');
grid on
axis tight


%Conclusion

% In Task 11, we extended our exploration of trajectory estimation with motion, 
% focusing on the influence of the regularization parameter μ on the quality 
% of our optimization results. Our primary objective was to investigate how the
% choice of μ affects the trade-off between Range Error (RE) and Velocity Error 
% (VE) within the context of trajectory optimization.
% 
% We initially generated a reference trajectory and introduced inconsistencies 
% by scaling down velocity measurements. This deliberate incongruity allowed 
% us to assess how different values of μ affect the optimization results. 
% Through a series of optimization processes for distinct μ values, we observed 
% that the resulting 2D plot of RE (x-axis) vs VE (y-axis) exhibits a distinct 
% hyperbolic shape.
% 
% 
% This plot's characteristics are insightful:
% 
% Inverse Relationship with μ: As μ increases, the Velocity Error (VE) tends to decrease 
% while the Range Error (RE) simultaneously increases. Conversely, smaller μ 
% values lead to higher VE and lower RE. This relationship signifies that the 
% choice of μ strongly influences the optimization results.
% 
% Trade-off between Error Types: The hyperbolic shape indicates a clear trade-off 
% between VE and RE. When optimizing with greater μ, the optimization process 
% places greater emphasis on satisfying the velocity constraints, 
% consequently yielding lower VE but higher RE. Conversely, smaller μ values 
% prioritize the range measurements, yielding lower RE but higher VE.
% 
% This observation aligns with our previous findings in Task 10, where we found 
% that smaller μ values make the velocity requirements less critical, while 
% larger μ values prioritize velocity constraints over range constraints.

%Note: Proving RE vs VE is decreasing is in a pdf file in GitHub


%% Task 12

close all;
clear;
clc;

figure;
xlim([-21, 21]);
ylim([-21, 21]);
grid on;
hold on;
xlabel('X (meters)');
ylabel('Y (meters)');
title('Generated 2D Trajectory with Anchors');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the bounding box and scaling factors
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
disp('Press Space key to end the input.');
disp(' ');
k = 1;


while 1
    [x(k), y(k), button] = ginput(1);

    if button == 32 % 32 = Space key
        x(k) = [];  % delete last point acquired when space is pressed
        y(k) = [];  % delete last point acquired when space is pressed
        break
    end
    
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

npt = length(x);        % number of via points, including initial and final
nvia = 0:1:npt-1;

csinterp_x = csapi(nvia, x);
csinterp_y = csapi(nvia, y);

T = 75; %number of samples
time = linspace(0, npt-1, T); 
xx = fnval(csinterp_x, time);
yy = fnval(csinterp_y, time);

% Plot the generated trajectory ('real one')
plot(xx, yy, 'ro-');

% Simulate target motion and record measurements
sample_rate = 2; % Hz
dt = 1 / sample_rate;
trajectory = [xx; yy];

% Initialize arrays to store measurements
ranges = zeros(T, 4); % Four anchors
angles = zeros(T, 4); % Four anchors
velocities = zeros(2, T);

for t = 1:T
    % Simulate target motion
    if t == 1
        delta_position = trajectory(:, t+1) - trajectory(:, t); %initial speed
        velocity = (delta_position) / dt; 
    elseif t == T 
        delta_position = trajectory(:, t-1) - trajectory(:, t); %final speed
        velocity = (delta_position) / dt;  
    else
        % Compute velocity from consecutive positions
        delta_position = trajectory(:, t+1) - trajectory(:, t-1); %formula 3 do enunciado
        velocity = (delta_position) / (2*dt);
    end
    
    velocities(:, t) = 1*velocity; %Set of velocity vectors (0.8 introduces an inconsistency)
    
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add Gaussian white noise to range and velocity measurements
std_dev_range = 0.1; % Standard deviation for range measurements (0.1m)
std_dev_velocity = 0.1 / sqrt(2); % Standard deviation for velocity measurements (0.1/√2 m/s)

noisy_ranges = ranges + std_dev_range * randn(T, 4);
noisy_velocities = velocities + std_dev_velocity * randn(2, T);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define optimization variables and find optimal trajectory for obtained
% measurements

disp(ranges)
mu = 0; % Value of mu. We gonna test it for mu=0 and mu=1

while mu < 2
    cvx_begin quiet
        variable x(2, T)

        % Define the cost function
        cost = 0;
        for t = 1:T

            %Range measurements part
            for anchor_idx = 1:4
                anchor_position = anchors(anchor_idx, :);
                delta = x(:, t) - anchor_position';
                range = norm(delta);
                cost = cost + square_pos(range - noisy_ranges(t, anchor_idx));
            end

            %Velocities part
            if t == T
                veloc = (x(:, t-1) - x(:, t))/dt;
            elseif t == 1
                veloc = (x(:, t+1) - x(:, t))/dt;
            else
                veloc = (x(:, t+1) - x(:, t-1))/(2*dt);
            end

            cost = cost + mu * square_pos(norm(veloc - noisy_velocities(:, t)));
        end

        minimize(cost)  
    cvx_end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plot the optimizated trajectory 
    figure;
    xlim([-21, 21]);
    ylim([-21, 21]);
    grid on;
    hold on;
    xlabel('X (meters)');
    ylabel('Y (meters)');
    title(['Optimization 2D Trajectory for \mu = ', num2str(mu)]);
    plot(anchors(:, 1), anchors(:, 2), 'bo', 'MarkerSize', 7, 'MarkerFaceColor', 'b');
    rectangle('Position', [x_min, y_min, x_max - x_min, y_max - y_min], 'LineWidth', 0.5);
    plot(xx, yy, 'ro-'); %Real Trajectory


    for t = 1:T
        plot(x(1, t), x(2, t), 'bx'); %Optimized trajectory
    end
    
    % Compute the global mean navigation error (MNE)
    mean_navigation_error = 0;
    for t = 1:T
        error_at_t = norm(x(:, t) - trajectory(:, t));
        mean_navigation_error = mean_navigation_error + error_at_t;
    end
    mean_navigation_error = (1 / T) * mean_navigation_error;

    % Display the global mean navigation error
    fprintf('\nGlobal Mean Navigation Error (MNE)for u = %.0f: %f\n', mu, mean_navigation_error);

    mu = mu + 1;
end


% % Conclusion
% 
% Global Mean Navigation Error (MNE)for u = 0: 0.091678
% 
% Global Mean Navigation Error (MNE)for u = 1: 0.064157


% Is the error distributed uniformly across the trajectories?
% 
% Indeed, it appears that the error is distributed relatively uniformly across 
% the trajectories. This uniform distribution of error implies that the 
% optimization variables (position and velocity) do not exhibit significant 
% deviations from their real trajectory counterparts. The errors introduced 
% in both range and velocity measurements seem to be distributed consistently
% over the entire trajectory. This behavior is indicative of a well-balanced 
% optimization process.
% 
% Furthermore, when the regularization parameter μ is set to 1, the velocity 
% measurements play a significant role in improving the trajectory accuracy. 
% This inclusion of velocity information allows the optimization to produce 
% a more accurate trajectory. Consequently, the global mean navigation error 
% (MNE) is smaller when μ is equal to 1, indicating a more faithful 
% representation of the real trajectory. This result suggests that 
% incorporating velocity measurements, when available, can lead to better 
% trajectory estimation.
% 
% 
% 
% What happens as you reduce the number of anchors?
% 
% When the number of anchors is reduced, it has the potential to impact the 
% uniform distribution of error across the trajectory. With fewer anchors, 
% there are fewer measurements available for the optimization process. As a 
% result, the errors induced in each measurement can have a more pronounced 
% effect on the trajectory estimation. The smaller number of measurements 
% reduces the ability to "cancel out" or "balance" these errors. This is 
% particularly noticeable in scenarios where the measurements' errors, both 
% in range and velocity, are significant.
% 
% In the context of fewer anchors, the optimization process may encounter 
% greater difficulty in producing a trajectory that closely matches the real 
% trajectory. The decreased redundancy in measurements could result in a trajectory 
% that deviates more from the actual motion of the target. Therefore, 
% in such scenarios, the error distribution across the trajectory may be 
% less uniform compared to cases with a greater number of anchors.
% 
% In conclusion, the uniform distribution of error in trajectory estimation 
% is influenced by factors such as the value of μ and the number of available 
% anchor measurements. Smaller values of μ, especially when incorporating 
% velocity data, tend to lead to more accurate trajectory estimates. 
% Conversely, reducing the number of anchors may disrupt the uniformity 
% of error distribution, potentially causing the trajectory to deviate 
% further from the real motion of the target. These insights emphasize the 
% importance of measurement redundancy and appropriate regularization in 
% improving trajectory estimation accuracy.


%% TASK 13
% Define parameters
clear;
close all;
clc;

% Define the bounding box
x_min = -20;  % Minimum x-coordinate
x_max = 20;   % Maximum x-coordinate
y_min = -20;  % Minimum y-coordinate
y_max = 20;   % Maximum y-coordinate

% Plot the bounding box
rectangle('Position', [x_min, y_min, x_max - x_min, y_max - y_min], 'EdgeColor', 'r', 'LineWidth', 2);


% Define the single anchor position
anchor_x = rand * (x_max - x_min) + x_min;  % Randomly place the anchor within the bounding box
anchor_y = 20;  % Place the anchor at y = 20

anchor = [anchor_x, anchor_y];  % Create a 2D coordinate for the anchor

% Plot the single anchor
plot(anchor_x, anchor_y, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
hold on 

% Illustrate the bounding box
x_in = [x_min, x_max, x_max, x_min, x_min];
y_in = [y_min, y_min, y_max, y_max, y_min];
plot(x_in, y_in, 'b-', 'LineWidth', 2);

% Set axis limits to match the bounding box
axis([x_min - 5, x_max + 5, y_min - 5, y_max + 5]);

% Generate the trajectory
disp('Use the mouse to input via points for the reference trajectory');
disp('Press the Space key to end the input');
button = 1;
k = 1;

while 1 
    [x(k), y(k), button] = ginput(1);
    if button == 32 % 32 = Space key
        x(k) = [];  % delete last point acquired when space is pressed
        y(k) = [];  % delete last point acquired when space is pressed
        break
    end
    plot(x(k), y(k), 'r+');
    k = k + 1;
end

% If more than two points are selected
if k-1 > 2 
    disp('ERROR: Maximum of 2 points for a linear trajectory');
    close all;
    return
end

drawnow;
disp([num2str(k - 1), ' points to interpolate from']);

% Generating the trajectory
npt = length(x);        % Number of via points, including initial and final
nvia = 0:1:npt - 1;     % Array to store the indices of the via points 
csinterp_x = csapi(nvia, x);    % Smooth lines that connect the via points 
csinterp_y = csapi(nvia, y);     
time = linspace(0, npt - 1, 50);   % Generate 50 points from 0 to 2
xx = fnval(csinterp_x, time);
yy = fnval(csinterp_y, time);

x0 = [xx(1),yy(1)]; % Initial position

plot(xx, yy, 'ro');
saveas(gcf,"Task13.png")

% Simulate target motion and record measurements
T = length(time); %number of samples
sample_rate = 4; % Hz
dt = 1 / sample_rate;
trajectory = [xx; yy];

% Initialize arrays to store measurements
ranges = zeros(T, 1); % Now only 1 anchor
range_rates = zeros(T, 1);
angles = zeros(T, 1);
directions = zeros(2,T);

% Compute Velocity - Constant in a linear trajectory
velocity = [(xx(end) - xx(1))/(T*dt),(yy(end) - yy(1))/(T*dt)];

for t = 1:T       
    % Compute range and angle measurements for one anchor    
    target_position = trajectory(:, t)';
    delta = target_position - anchor;
    range = norm(delta);
    angle = atan2(delta(2), delta(1));
    ranges(t, 1) = range;
    
    %Just to check if the measurements are ok
    angles(t, 1) = rad2deg(angle);  
    
    %Compute the direction from the anchor pointing towards the target
    directions(:,t) = [cos(angle), sin(angle)];  %Angle already in radians
    range_rates(t)= dot(velocity,directions(:, t));        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add Gaussian white noise to range and range rate measurements
std_dev_range = 0.1; % Standard deviation for range measurements (0.1m)
std_dev_range_rates = 0.1/sqrt(2);

noisy_ranges = ranges + std_dev_range * randn(T, 1);
noisy_range_rates = range_rates + std_dev_range_rates * randn(T, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TASK 14 (run imediatelly after running task 13 please)
close all;

nu = 100;

% Define anchor locations 
a = anchor;

% Define a grid of x and y values for contour plot
vx = -3.5:0.1:3.5;
vy = -3.5:0.1:3.5;
[Vx, Vy] = meshgrid(vx, vy);

% Initialize the cost function matrix
cost = zeros(size(Vx));

% Calculate the cost function 
for i = 1:numel(Vx)
    v_current = [Vx(i), Vy(i)]; % Current velocity
    for t = 1:T
        rhat = norm(x0 + v_current.*t - a);
        shat = dot(v_current,x0 + v_current.*t - a)/rhat;
        cost(i) = cost(i) + (rhat - noisy_ranges(t))^2 + nu*(shat - noisy_range_rates(t))^2;
    end
end 

% Create a contour plot of the cost function
figure;
contour(Vy, Vx, cost, 1000); % Adjust the number of contour lines as needed
hold on;
plot(velocity(1),velocity(2), 'r.', 'MarkerSize', 20); % Plot Real velocity
xlabel('x');
ylabel('y');
title('Cost Function Contour Plot');
colorbar;
saveas(gcf,"Task14.png");
hold off;










