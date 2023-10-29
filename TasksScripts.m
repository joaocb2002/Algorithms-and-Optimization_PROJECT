% %   OPTIMIZATION AND ALGORITHMS 2023/2024
%  
% %   INSTITUTO SUPERIOR TÉCNICO MEEC
% 
% %   Author: João Castelo Branco, Miguel Lameiras, Rodrigo Faria e Bernardo Soares
% 
% %   This file contains the Matlab resolution of Tasks 1-16 of the course's
% %   project.

%% Task1 
close all;

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



%% Task2
close all;

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
surf(X,Y,cost)
hold on;
plot(a1(1), a1(2), 'r.', 'MarkerSize', 20); % Display an Anchor
plot(a2(1), a2(2), 'r.', 'MarkerSize', 20); % Display an Anchor
xlabel('x');
ylabel('y');
title('Cost Function Contour Plot');
colorbar;
saveas(gcf,"Task2.png");





%% Task3a - Repetition of task 1 with convex approximation
close all;

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
close all;

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



%% Task 4a 
close all;

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
close all;

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



%% Task 5

% Exercício teórico



%% Task 6

% Exercício teórico



%% Task 7a
% Scenario (i) - Single Anchor
close all;

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
close all;

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

disp("Solving Task 10...");
for i = 1:6
    disp("Computing Solution for mu = " + string(mu_values(i)) + "...");
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




%% Task 11
close all;

% Induce inconsistency
inconsistent_velocities = 0.8.*velocities;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define optimization variables and find optimal trajectory for obtained
% measurements and determine range and velocity error calculation

values = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000]; % Values of mu

RE = zeros(1, length(values));
VE = zeros(1, length(values));

disp("Solving Task 11...");
for k = 1:length(values)
    mu = values(k);
    disp("Computing Solution for mu = " + string(mu) + "...");
    
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

            cost = cost + mu * square_pos(norm(veloc - inconsistent_velocities(:, t)));
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

        VE(1, k) = (norm(veloc - inconsistent_velocities(:, t)))^2;

    end
end

% Plot the Range Error (RE) vs Velocity Error (VE)
figure;
semilogx(RE, VE, 'ro-');
title('2D plot for RE(\mu) vs VE(\mu)');
xlabel('Range Error (RE)');
ylabel('Velocity Error (VE)');
grid on;
axis tight;
saveas(gcf,"Task11.png")



%% Task 12

close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add Gaussian white noise to range and velocity measurements
std_dev_range = 0.1; % Standard deviation for range measurements (0.1m)
std_dev_velocity = 0.1 / sqrt(2); % Standard deviation for velocity measurements (0.1/√2 m/s)

noisy_ranges = ranges + std_dev_range * randn(T, 4);
noisy_velocities = velocities + std_dev_velocity * randn(2, T);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define optimization variables and find optimal trajectory for obtained
% measurements

mu = 0; % Value of mu. We gonna test it for mu=0 and mu=1

disp("Solving Task 12...");
while mu < 2
    cvx_begin quiet
        disp("Computing Solution for mu = " + string(mu) + "...");
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





%% Task 13

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
anchor_x = (x_max - x_min) + x_min;  % Randomly place the anchor within the bounding box
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

plot(xx, yy, 'mo');
saveas(gcf,"Task13.png")

% Simulate target motion and record measurements
T = length(time); %number of samples
sample_rate = 2; % Hz
dt = 1 / sample_rate;
trajectory = [xx; yy];

% Initialize arrays to store measurements
ranges = zeros(T, 1); % Now only 1 anchor
range_rates = zeros(T, 1);
angles = zeros(T, 1);
directions = zeros(2,T);

% Compute Velocity - Constant in a linear trajectory
velocity = [(xx(end) - xx(1))/(T),(yy(end) - yy(1))/(T)];

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

% Create a figure
figure;

% Subplot 1: Range Measurements
subplot(2, 1, 1);

% Plot the true and noisy range measurements
plot(time, ranges, 'b', 'DisplayName', 'Range (True)');
hold on;
plot(time, noisy_ranges, 'g', 'DisplayName', 'Range (Noisy)');
hold off;

title('Range Measurements');
xlabel('Time');
ylabel('Range');
legend;

% Subplot 2: Range Rate Measurements
subplot(2, 1, 2);

% Plot the true and noisy range rate measurements
plot(time, range_rates, 'r', 'DisplayName', 'Range Rate (True)');
hold on;
plot(time, noisy_range_rates, 'm', 'DisplayName', 'Range Rate (Noisy)');
hold off;

title('Range Rate Measurements');
xlabel('Time');
ylabel('Range Rate');
legend;


%% Task 14

close all;

nu = 1e3;

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
        shat = dot(v_current, x0 + v_current.*t - a)/ rhat;
        cost(i) = cost(i) + (rhat - noisy_ranges(t))^2 + nu*(shat - noisy_range_rates(t))^2;
    end
end 

% Create a contour plot of the cost function
figure;
contour(Vx, Vy, cost, 300); % Adjust the number of contour lines as needed
hold on;
plot(velocity(1),velocity(2), 'r.', 'MarkerSize', 20); % Plot Real velocity
xlabel('x');
ylabel('y');
title('Cost Function Contour Plot');
colorbar;
saveas(gcf,"Task14.png");
hold off;

%% Task 15

%Exercicio Teorico 

%% Task 16
close all;

% Initial velocity estimate
initial_velocity = [0, -1];

% Set the desired step size tolerance (adjust this value as needed)
func_tolerance = 1e-3;

% Set up options for the Levenberg-Marquardt algorithm
options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt', 'Display', 'iter', 'FunctionTolerance', func_tolerance, 'OutputFcn', @output_function);

% Define the objective funtion to minimize
objective_function = @(velocity) cost_function(velocity, x0, a, noisy_ranges, noisy_range_rates, nu);

% Use LM algorithm to find the optimized velocity
predicted_velocity = lsqnonlin(objective_function, initial_velocity, [], [], options);

% Display the optimized velocity
disp('Predicted Velocity:');
disp(predicted_velocity);

% Evaluate the cost function at the optimized parameters
minimum_cost = cost_function(predicted_velocity, x0, a, noisy_ranges, noisy_range_rates, nu);
disp(['Minimum Cost Value: ', num2str(minimum_cost)]);

% Define the cost funtion
function total_cost = cost_function(velocity, x0, a, noisy_ranges, noisy_range_rates, nu)
    % Initialize total cost
    total_cost = 0;
    
    % Loop over all time instants
    T = length(noisy_ranges);
    for t = 1:T
        % Calculate the estimated range and range rate
        rhat = norm(x0 + velocity * t - a);
        shat = dot(velocity, x0 + velocity * t - a) / rhat;
        
        % Compute the squared differences and add them to the total cost
        cost_t = (rhat - noisy_ranges(t))^2 + nu * (shat - noisy_range_rates(t))^2;
        total_cost = total_cost + cost_t;
    end
end

% Output function to collect optimization information
function stop = output_function(x, optimValues, state)
    persistent iter
    persistent cost_values
    persistent gradient_norm_values
    
    % Initialize data on the first iteration
    if isempty(iter)
        iter = [];
        cost_values = [];
        gradient_norm_values = [];
    end
    
    % Store iteration count, cost value, and gradient norm
    iter = [iter; optimValues.iteration];
    cost_values = [cost_values; optimValues.residual];
    gradient_norm_values = [gradient_norm_values; norm(optimValues.gradient)];
    
    % Check if the optimization has completed
    if strcmp(state, 'done')
        % Create a logarithmic y-axis plot for cost
        figure;
        subplot(2, 1, 1);
        semilogy(iter, cost_values, '-or');
        title('Convergence of Cost Function');
        xlabel('Iteration');
        ylabel('Cost Function');
        grid on
        
        % Create a logarithmic y-axis plot for gradient norm
        subplot(2, 1, 2);
        semilogy(iter, gradient_norm_values, '-o');
        title('Convergence of Gradient Norm');
        xlabel('Iteration');
        ylabel('Gradient Norm');
        grid on;
        
        stop = true; % Terminate optimization

        saveas(gcf,"Task16.png");
    else
        stop = false;
    end
end

%EOF
