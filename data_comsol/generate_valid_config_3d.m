function [xgoal,ygoal,zgoal,rgoal,obstacles_matrix,obstacles_shpaes,nb_obstacles] = generate_valid_config_3d(num_scene, data_path)
    %clear;
    clc;
    rng shuffle
    area_length             = 9;
    % limits of the form: [x, y, z, dx, dy, dz, shape]
    area_limits             = [[0, area_length, 0,  area_length, 0.25, area_length, 111];
                               [area_length, 0, 0,  0.25, area_length, area_length, 111];
                               [0, 0, area_length,  area_length, area_length, 0.25, 111];
                               [0, -0.25, 0,        area_length, 0.25, area_length, 111];
                               [-0.25, 0, 0,        0.25, area_length, area_length, 111];
                               [0, 0, -0.25,        area_length, area_length, 0.25, 111]];
    area_limits_shapes      = ones(1,6); % 4 rectangles (1)
    xgoal                   = 1 + (-1+area_length)*rand(1);
    ygoal                   = 1 + (-1+area_length)*rand(1);
    zgoal                   = 1 + (-1+area_length)*rand(1);
    dgoal                   = 0.18;
    rgoal                   = dgoal/2;
    goal_limits             = [xgoal, ygoal, zgoal, dgoal, dgoal, dgoal, 222];
    goal_shape              = 6;
    nb_obstacles            = randi ([0 10]);
    obstacles_shpaes        = [randi([1 5], 1,nb_obstacles), area_limits_shapes, goal_shape];
    obstacles_matrix        = [zeros(nb_obstacles,7); area_limits; goal_limits];
    obs_vector              = [0, 0, 0, 0, 0, 0];


    for ind = 1:nb_obstacles
        INTERS     = ones(nb_obstacles,1);
        while sum(INTERS(:))
                rng shuffle
                x_obs       = 1 + (-1+area_length)*rand(1);
                y_obs       = 1 + (-1+area_length)*rand(1);
                z_obs       = 1 + (-1+area_length)*rand(1);
                width       = (1 + (-1+area_length)*rand(1))*0.3;
                height      = (1 + (-1+area_length)*rand(1))*0.3;
                depth       = (1 + (-1+area_length)*rand(1))*0.3;
                obs_vector  = [x_obs, y_obs, z_obs, width, height, depth];
                INTERS      = cubeint(obstacles_matrix,obs_vector);
        end
        obstacles_matrix(ind,7) = obstacles_shpaes(ind);
        for i = 1:6
            obstacles_matrix(ind,i)= obs_vector(i);  
        end
       
    end
    dlmwrite(fullfile(data_path, strcat('scene_',num2str(num_scene),'.txt') ),obstacles_matrix);
end
