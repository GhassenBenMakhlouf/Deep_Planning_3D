import com.comsol.model.*
import com.comsol.model.util.*


model_path = './';
data_path = './data/';
number_generated_data = 10;

for num_config = 1:number_generated_data
    tic
    
    model = ModelUtil.create('Model');
    
    model.modelPath(model_path);
    
    model.label('deep_planning_3d.mph');
    
    model.param.set('mu', '12.56e-7');
    model.param.set('sigmae', '100');
    model.param.set('sigmag', '1e6');
    model.param.set('sigmao', '0');
    model.param.set('w', '0.05');
    model.param.set('ACe', 'mu*sigmae*w*i');
    model.param.set('ACg', 'mu*sigmag*w*i');
    model.param.set('ACo', 'mu*sigmao*w*i');
    
    shape.block = 1; shape.cube = 2; shape.T = 3; shape.U = 4; shape.L = 5; shape.sphere = 6; shape.ellipsoid = 7;
    % 2D version: 
    % shape.rect = 1; shape.square = 2; shape.T = 3; shape.U = 4; shape.L = 5; shape.circle = 6; shape.ellipse =7;
    
    [xgoal,ygoal,zgoal,rgoal,obstacles_matrix,obstacles_shpaes,nb_obstacles] = generate_valid_config_3d(num_config, data_path);
    
    %%%%%%%%%%%%%%%%% print configuration parameters %%%%%%%%
    fprintf('configuration num %d :\n',num_config);
    xgoal
    ygoal
    zgoal
    rgoal
    obstacles_matrix
    obstacles_shpaes
    nb_obstacles
    
    
    model.param.set('x_goal', num2str(xgoal));
    model.param.set('y_goal', num2str(ygoal));
    model.param.set('z_goal', num2str(zgoal));
    
    model.component.create('comp1', false);
    
    model.component('comp1').geom.create('geom1', 3);
    
    model.result.table.create('evl3', 'Table');
    
    model.component('comp1').mesh.create('mesh1');
    
    %%%%%%%% Build the environment block %%%%%%%%%
    model.component('comp1').geom('geom1').create('envir', 'Block');
    model.component('comp1').geom('geom1').feature('envir').label('Block 1');
    model.component('comp1').geom('geom1').feature('envir').set('selresult', true);
    model.component('comp1').geom('geom1').feature('envir').set('selresultshow', 'all');
    model.component('comp1').geom('geom1').feature('envir').set('size', [10 10 10]);
    
    %%%%%%%% Build the Goal Geometry %%%%%%%%%
    model.component('comp1').geom('geom1').create('goal', 'Sphere');
    model.component('comp1').geom('geom1').feature('goal').set('selresult', true);
    model.component('comp1').geom('geom1').feature('goal').set('selresultshow', 'all');
    model.component('comp1').geom('geom1').feature('goal').set('r', num2str(rgoal));
    model.component('comp1').geom('geom1').feature('goal').set('pos', {'x_goal' 'y_goal' 'z_goal'});
    
    %%%%%%%% Build Obstacles Geometries %%%%%%%%%
    labels= Build_obstacles(model, obstacles_matrix, nb_obstacles, obstacles_shpaes, shape);
    
    %%%%%%%% Cut Obstacles Geometries from Environment %%%%%%%%%
    model.component('comp1').geom('geom1').create('dif1', 'Difference');
    model.component('comp1').geom('geom1').feature('dif1').selection('input').set({'goal' 'envir'});
    if nb_obstacles > 0
        model.component('comp1').geom('geom1').feature('dif1').selection('input2').set(labels);
    else
        continue;
    end
    
    model.component('comp1').geom('geom1').run;
    
    index_env_edge1 = mphselectbox(model, 'geom1', [0 10; 0 10; -1 0], 'boundary');
    index_env_edge2 = mphselectbox(model, 'geom1', [-1 0; 0 10; 0 10], 'boundary');
    index_env_edge3 = mphselectbox(model, 'geom1', [0 10; -1 0; 0 10], 'boundary');
    index_env_edge4 = mphselectbox(model, 'geom1', [0 10; 0 10; 10 11], 'boundary');
    index_env_edge5 = mphselectbox(model, 'geom1', [10 11; 0 10; 0 10], 'boundary');
    index_env_edge6 = mphselectbox(model, 'geom1', [0 10; 10 11; 0 10], 'boundary');
    index_env = cat(2, index_env_edge1, index_env_edge2, index_env_edge3, index_env_edge4, index_env_edge5, index_env_edge6);
    model.component('comp1').selection.create('sel1', 'Explicit');
    model.component('comp1').selection('sel1').geom('geom1', 2); %3 for domains, 2 for boundaries/domains,1...
    %for edges/boundaries, and 0 for points.
    model.component('comp1').selection('sel1').set(index_env);
    
    model.material.create('mat1', 'Common', '');
    model.material('mat1').propertyGroup('def').func.create('an1', 'Analytic');
    model.material('mat1').label('env');
    model.material('mat1').propertyGroup('def').func('an1').set('expr', 'sigmae*1/sqrt((x-x_goal)^2+ (y-y_goal)^2+ (z-z_goal)^2)');
    model.material('mat1').propertyGroup('def').func('an1').set('args', {'x' 'y' 'z'});
    model.material('mat1').propertyGroup('def').func('an1').set('argunit', '1,1,1');
    model.material('mat1').propertyGroup('def').func('an1').set('plotargs', {'x' '0' '10'; 'y' '0' '10'; 'z' '0' '10'});
    model.material('mat1').propertyGroup('def').set('electricconductivity', {'an1(x,y,z)'});
    
    model.component('comp1').physics.create('hzeq', 'HelmholtzEquation', 'geom1');
    model.component('comp1').physics('hzeq').field('dimensionless').field('B');
    model.component('comp1').physics('hzeq').prop('Units').set('DependentVariableQuantity', 'magneticfluxdensity');
    model.component('comp1').physics('hzeq').prop('Units').set('SourceTermQuantity', 'currentdensity');
    model.component('comp1').physics('hzeq').feature('heq1').set('c', -1);
    model.component('comp1').physics('hzeq').feature('heq1').set('a', 'if(dom==1,mu*w*i*mat1.def.an1(x,y,z),if((dom==2),ACg,ACo))');
    model.component('comp1').physics('hzeq').feature('heq1').set('f', 1);
    model.component('comp1').physics('hzeq').feature('init1').set('B', 'if((dom==1||dom==2),0,1)');
    model.component('comp1').physics('hzeq').feature('init1').set('Bt', 'if((dom==1||dom==2),0,0)');
    
    % FreeTri does not fill up the volume
    % model.component('comp1').mesh('mesh1').create('ftri1', 'FreeTri');
    model.component('comp1').mesh('mesh1').create('ftri1', 'FreeTet');
    model.component('comp1').mesh('mesh1').feature('size').set('hauto', 3);
    model.component('comp1').mesh('mesh1').feature('size').set('custom', 'on');
    model.component('comp1').mesh('mesh1').feature('size').set('hmax', 0.5);
    model.component('comp1').mesh('mesh1').feature('size').set('hmin', 0.3);
    model.component('comp1').mesh('mesh1').feature('ftri1').selection.remaining;
    model.component('comp1').mesh('mesh1').run;
    
    model.result.table('evl3').label('Evaluation 3D');
    model.result.table('evl3').comments('Interactive 3D values');
    
    model.capeopen.label('Thermodynamics Package');
    
    model.study.create('std1');
    model.study('std1').create('freq', 'Frequency');
    model.study('std1').feature('freq').set('activate', {'hzeq' 'on'});
    
    model.sol.create('sol1');
    model.sol('sol1').study('std1');
    model.sol('sol1').attach('std1');
    model.sol('sol1').create('st1', 'StudyStep');
    model.sol('sol1').create('v1', 'Variables');
    model.sol('sol1').create('s1', 'Stationary');
    model.sol('sol1').feature('s1').create('p1', 'Parametric');
    model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
    model.sol('sol1').feature('s1').feature.remove('fcDef');
    
    model.study('std1').feature('freq').set('plist', 0.05);
    model.study('std1').feature('freq').set('preusesol', 'yes');
    model.study('std1').feature('freq').set('discretization', {'hzeq' 'physics'});
    
    model.sol('sol1').attach('std1');
    model.sol('sol1').feature('v1').set('clistctrl', {'p1'});
    model.sol('sol1').feature('v1').set('cname', {'freq'});
    model.sol('sol1').feature('v1').set('clist', {'0.05[Hz]'});
    model.sol('sol1').feature('s1').feature('p1').set('pname', {'freq'});
    model.sol('sol1').feature('s1').feature('p1').set('plistarr', [0.05]);
    model.sol('sol1').feature('s1').feature('p1').set('punit', {'Hz'});
    model.sol('sol1').feature('s1').feature('p1').set('pcontinuationmode', 'no');
    model.sol('sol1').feature('s1').feature('p1').set('preusesol', 'yes');
    model.sol('sol1').runAll;
    toc
    
    
    % added through comsol gui
    model.result.dataset.create('an1_ds1', 'Grid3D');
    model.result.dataset('an1_ds1').set('source', 'data');
    model.result.dataset('an1_ds1').set('parmax1', 10);
    model.result.dataset('an1_ds1').set('parmax2', 10);
    model.result.dataset('an1_ds1').set('parmax3', 10);
    model.result.dataset('an1_ds1').set('source', 'function');
    model.result.dataset('an1_ds1').set('functionlist', 'material/mat1/def');
    model.result.dataset('an1_ds1').set('function', 'an1');
    model.result.dataset.create('an1_ds2', 'Grid3D');
    model.result.dataset('an1_ds2').set('parmax1', 10);
    model.result.dataset('an1_ds2').set('parmax2', 10);
    model.result.dataset('an1_ds2').set('parmax3', 10);
    model.result.dataset('an1_ds2').set('source', 'function');
    model.result.dataset('an1_ds2').set('functionlist', 'material/mat1/def');
    model.result.dataset('an1_ds2').set('function', 'an1');
    model.result.dataset.create('cln1', 'CutLine3D');
    model.result.dataset('cln1').set('genpoints', {'3' '3' '3'; 'x_goal' 'y_goal' 'z_goal'});
    
    model.result.create('pg1', 'PlotGroup3D');
    
%     model.result('pg1').create('mfm', 'Volume');
%     model.result('pg1').feature('mfm').set('smooth', 'internal');
%     model.result('pg1').feature('mfm').set('resolution', 'normal');
%     model.result('pg1').feature('mfm').set('resolution', 'custom');
%     model.result('pg1').feature('mfm').set('refine', 5);
    
%     model.result('pg1').create('mslc1', 'Multislice');
%     model.result('pg1').feature('mslc1').set('xcoord', 'range(0,0.1,10)');
%     model.result('pg1').feature('mslc1').set('ycoord', 'range(0,0.1,10)');
%     model.result('pg1').feature('mslc1').set('zcoord', 'range(0,0.1,10)');
 
    model.result('pg1').create('scv1', 'ScatterVolume');
    model.result('pg1').feature('scv1').set('expr', {'x' 'y' 'z'});
    model.result('pg1').feature('scv1').set('arrowxmethod', 'coord');
    model.result('pg1').feature('scv1').set('xcoord', 'range(0,0.1,10)');
    model.result('pg1').feature('scv1').set('arrowymethod', 'coord');
    model.result('pg1').feature('scv1').set('ycoord', 'range(0,0.1,10)');
    model.result('pg1').feature('scv1').set('arrowzmethod', 'coord');
    model.result('pg1').feature('scv1').set('zcoord', 'range(0,0.1,10)');
    model.result('pg1').feature('scv1').set('sphereradiusscaleactive', true);
    model.result('pg1').feature('scv1').set('sphereradiusscale', 0.1);

    model.result('pg1').create('str1', 'Streamline');
    model.result('pg1').feature('str1').set('posmethod', 'magnitude');
    model.result('pg1').feature('str1').set('madv', 'manual');
    model.result('pg1').feature('str1').set('smooth', 'internal');
    model.result('pg1').feature('str1').set('resolution', 'normal');
    model.result('pg1').feature('str1').set('color', 'black');
    model.result('pg1').create('arwv1', 'ArrowVolume');
    model.result('pg1').feature('arwv1').set('xcoord', 'range(0,0.1,10)');
    model.result('pg1').feature('arwv1').set('ycoord', 'range(0,0.1,10)');
    model.result('pg1').feature('arwv1').set('zcoord', 'range(0,0.1,10)');
    model.result('pg1').feature('arwv1').set('scaleactive', true);
    model.result('pg1').feature('arwv1').set('scale', '0.00005');
    
    model.result.create('pg2', 'PlotGroup3D');
    model.result('pg2').create('vol1', 'Volume');
    model.result('pg2').set('data', 'none');
    model.result('pg2').set('titletype', 'manual');
    model.result('pg2').set('title', 'an1(x,y,z)');
    model.result('pg2').set('edges', false);
    model.result('pg2').feature('vol1').set('data', 'an1_ds2');
    model.result('pg2').feature('vol1').set('solrepresentation', 'solnum');
    model.result('pg2').feature('vol1').set('expr', 'mat1.def.an1(root.x[1],root.y[1],root.z[1])');
    model.result('pg2').feature('vol1').set('unit', '');
    model.result('pg2').feature('vol1').set('descractive', true);
    model.result('pg2').feature('vol1').set('descr', 'conductivity(x,y)');
    model.result('pg2').feature('vol1').set('titletype', 'custom');
    model.result('pg2').feature('vol1').set('typeintitle', false);
    model.result('pg2').feature('vol1').set('unitintitle', false);
    model.result('pg2').feature('vol1').set('rangecoloractive', true);
    model.result('pg2').feature('vol1').set('rangecolormin', 15.713484026367723);
    model.result('pg2').feature('vol1').set('rangecolormax', 200);
    model.result('pg2').feature('vol1').set('rangedataactive', true);
    model.result('pg2').feature('vol1').set('rangedatamin', 0);
    model.result('pg2').feature('vol1').set('rangedatamax', 14000);
    model.result('pg2').feature('vol1').set('smooth', 'none');
    model.result('pg2').feature('vol1').set('allowmaterialsmoothing', false);
    model.result('pg2').feature('vol1').set('resolution', 'normal');
    
    model.result.create('pg3', 'PlotGroup1D');
    model.result('pg3').create('lngr1', 'LineGraph');
    model.result('pg3').set('data', 'cln1');
    model.result('pg3').set('xlabel', 'Arc length');
    model.result('pg3').set('ylabel', 'Dependent variable B (T)');
    model.result('pg3').set('window', 'window1');
    model.result('pg3').set('windowtitle', 'Plot 1');
    model.result('pg3').set('xlabelactive', false);
    model.result('pg3').set('ylabelactive', false);
    model.result('pg3').feature('lngr1').set('smooth', 'internal');
    model.result('pg3').feature('lngr1').set('resolution', 'normal');
    model.result.export.create('plot1', 'Plot');
    model.result.export.create('plot2', 'Plot');
    model.result.export.create('plot3', 'Plot');
    model.result.export('plot1').set('plot', 'str1');
%     model.result.export('plot2').set('plot', 'mfm');
%     model.result.export('plot2').set('plot', 'mslc1');
    model.result.export('plot2').set('plot', 'scv1');
    model.result.export('plot2').set('sort', true);
    model.result.export('plot3').set('plot', 'arwv1');
    model.result.export('plot1').set('filename', strcat(data_path, 'streamline_C'...
        ,num2str(num_config),'.txt'));
    model.result.export('plot2').set('filename', strcat(data_path, 'magneticfield_C'...
        ,num2str(num_config),'.txt'));
    model.result.export('plot3').set('filename', strcat(data_path, 'gradient_C'...
        ,num2str(num_config),'.txt'));
    
    % model.result.export('plot1').run;
    model.result.export('plot2').run;
    % model.result.export('plot3').run;
    mphsave(model,'deep_planning_3d.mph');
end


function [labels]=Build_obstacles(model, obstacles_matrix, nb_obstacles, obstacles_shpaes, shape)
    labels = cell(1,nb_obstacles);
    % b--> block, s--> sphere, e--> ellipsoid, t--> T form, u--> U form, c--> cube, l--> L form 
    b=0;s=0;e=0;t=0;u=0;c=0;l=0;
    for obs_index =1:nb_obstacles
        if obstacles_shpaes(obs_index) == shape.block
            b = b + 1;
            label = strcat('b', num2str(b));
            model.component('comp1').geom('geom1').create(label, 'Block');
            model.component('comp1').geom('geom1').feature(label).set('pos',...
                [obstacles_matrix(obs_index,1) obstacles_matrix(obs_index,2), obstacles_matrix(obs_index,3)]);
            model.component('comp1').geom('geom1').feature(label).set('size',...
                [obstacles_matrix(obs_index,4) obstacles_matrix(obs_index,5) obstacles_matrix(obs_index,6)]);
            %(x-x0).^2+(y-y0).^2<=R^2)
            labels{obs_index}=label;
        elseif obstacles_shpaes(obs_index) == shape.sphere
            s = s + 1;
            label = strcat('sp', num2str(s));
            
            radius = min(obstacles_matrix(obs_index,4:6))/2;              
           
            x_center = obstacles_matrix(obs_index,1)+ radius;
            y_center = obstacles_matrix(obs_index,2)+ radius;
            z_center = obstacles_matrix(obs_index,3)+ radius;
        
            model.component('comp1').geom('geom1').create(label, 'Sphere');
            model.component('comp1').geom('geom1').feature(label).set('pos', [x_center y_center z_center]);
            model.component('comp1').geom('geom1').feature(label).set('r', radius);
            labels{obs_index}=label;
        elseif obstacles_shpaes(obs_index) == shape.ellipsoid
            e = e + 1;
            label = strcat('e', num2str(e));
            radius_w = obstacles_matrix(obs_index,4)/2;
            radius_h = obstacles_matrix(obs_index,5)/2;
            radius_d = obstacles_matrix(obs_index,6)/2;
            x_center = obstacles_matrix(obs_index,1)+radius_w;
            y_center = obstacles_matrix(obs_index,2)+radius_h;
            z_center = obstacles_matrix(obs_index,3)+radius_d;
            model.component('comp1').geom('geom1').create(label, 'Ellipsoid');
            model.component('comp1').geom('geom1').feature(label).set('pos', [x_center y_center z_center]);
            model.component('comp1').geom('geom1').feature(label).set('semiaxes', [radius_w radius_h radius_d]);
            labels{obs_index}=label;
        elseif obstacles_shpaes(obs_index) == shape.T
            t = t + 1;
            label = strcat('t', num2str(t));
            width_rec1 = obstacles_matrix(obs_index,4);
            height_rec1 = obstacles_matrix(obs_index,5)*(2/3);
            depth_rec1 = obstacles_matrix(obs_index,6);
        
            width_rec2 = obstacles_matrix(obs_index,4)*(2/3);
            height_rec2 = obstacles_matrix(obs_index,5)- height_rec1;
            depth_rec2 = obstacles_matrix(obs_index,6);
        
            x_rec1 = obstacles_matrix(obs_index,1);
            y_rec1 = obstacles_matrix(obs_index,2)+ (obstacles_matrix(obs_index,4)-height_rec1);
            z_rec1 = obstacles_matrix(obs_index,3);

            x_rec2 = obstacles_matrix(obs_index,1) + ((obstacles_matrix(obs_index,3)/2)-width_rec2/2);
            y_rec2 = obstacles_matrix(obs_index,2);
            z_rec2 = obstacles_matrix(obs_index,3);
                        
            model.component('comp1').geom('geom1').create(strcat(label,'1'), 'Block');
            model.component('comp1').geom('geom1').feature(strcat(label,'1')).set('pos', [x_rec1 y_rec1 z_rec1]);
            model.component('comp1').geom('geom1').feature(strcat(label,'1')).set('size', [width_rec1 height_rec1 depth_rec1]);
            model.component('comp1').geom('geom1').create(strcat(label,'2'), 'Block');
            model.component('comp1').geom('geom1').feature(strcat(label,'2')).set('pos', [x_rec2 y_rec2 z_rec2]);
            model.component('comp1').geom('geom1').feature(strcat(label,'2')).set('size', [width_rec2 height_rec2 depth_rec2]);

            model.component('comp1').geom('geom1').create(strcat(label,'uni1'), 'Union');
            model.component('comp1').geom('geom1').feature(strcat(label,'uni1')).selection('input').set({strcat(label,'1')...
                strcat(label,'2')});
            labels{obs_index}=strcat(label,'uni1');
        elseif obstacles_shpaes(obs_index) == shape.U
            u = u + 1;
            label = strcat('u', num2str(u));
            
            width_rec1 = obstacles_matrix(obs_index,4);
            height_rec1 = obstacles_matrix(obs_index,5)*(2/3);
            depth_rec1 = obstacles_matrix(obs_index,6);

            width_rec2 = obstacles_matrix(obs_index,4)*(1/3);
            height_rec2 = obstacles_matrix(obs_index,5)- height_rec1;
            depth_rec2 = obstacles_matrix(obs_index,6);
        
            width_rec3 = obstacles_matrix(obs_index,4)*(1/3);
            height_rec3 = obstacles_matrix(obs_index,5)- height_rec1;
            depth_rec3 = obstacles_matrix(obs_index,6);

            x_rec1 = obstacles_matrix(obs_index,1);
            y_rec1 = obstacles_matrix(obs_index,2);
            z_rec1 = obstacles_matrix(obs_index,3);

            x_rec2 = obstacles_matrix(obs_index,1);
            y_rec2 = obstacles_matrix(obs_index,2)+ height_rec1;
            z_rec2 = obstacles_matrix(obs_index,3);
        
            x_rec3 = obstacles_matrix(obs_index,1) + (obstacles_matrix(obs_index,3)*(1/3));
            y_rec3 = obstacles_matrix(obs_index,2) + height_rec1;
            z_rec3 = obstacles_matrix(obs_index,3);

            model.component('comp1').geom('geom1').create(strcat(label,'1'), 'Block');
            model.component('comp1').geom('geom1').feature(strcat(label,'1')).set('pos', [x_rec1 y_rec1 z_rec1]);
            model.component('comp1').geom('geom1').feature(strcat(label,'1')).set('size', [width_rec1 height_rec1 depth_rec1]);
            model.component('comp1').geom('geom1').create(strcat(label,'2'), 'Block');
            model.component('comp1').geom('geom1').feature(strcat(label,'2')).set('pos', [x_rec2 y_rec2 z_rec2]);
            model.component('comp1').geom('geom1').feature(strcat(label,'2')).set('size', [width_rec2 height_rec2 depth_rec2]);
            model.component('comp1').geom('geom1').create(strcat(label,'3'), 'Block');
            model.component('comp1').geom('geom1').feature(strcat(label,'3')).set('pos', [x_rec3 y_rec3 z_rec3]);
            model.component('comp1').geom('geom1').feature(strcat(label,'3')).set('size', [width_rec3 height_rec3 depth_rec3]);

            model.component('comp1').geom('geom1').create(strcat(label,'uni1'), 'Union');
            model.component('comp1').geom('geom1').feature(strcat(label,'uni1')).selection('input').set({strcat(label,'1')...
                strcat(label,'2') strcat(label,'3')});
            labels{obs_index}=strcat(label,'uni1');
            %model.component('comp1').geom('geom1').create('rot1', 'Rotate');
            %model.component('comp1').geom('geom1').feature('rot1').set('rot', 10);
            %model.component('comp1').geom('geom1').feature('rot1').set('pos', [5 5]);
            %model.component('comp1').geom('geom1').feature('rot1').selection('input').set({'uni1'});
            
        elseif obstacles_shpaes(obs_index) == shape.cube
            c = c + 1;
            label = strcat('c', num2str(c));
            model.component('comp1').geom('geom1').create(label, 'Block');
            model.component('comp1').geom('geom1').feature(label).set('pos', [obstacles_matrix(obs_index,1)...
                obstacles_matrix(obs_index,2) obstacles_matrix(obs_index,3)]);
            model.component('comp1').geom('geom1').feature(label).set('size', [obstacles_matrix(obs_index,4)...
                obstacles_matrix(obs_index,4) obstacles_matrix(obs_index,4)]);
            labels{obs_index}=label;
        elseif obstacles_shpaes(obs_index) == shape.L
            l = l + 1;
            label = strcat('l', num2str(l));
            height_rec1 = obstacles_matrix(obs_index,4);
            width_rec1 = obstacles_matrix(obs_index,5)*(2/3);
            depth_rec1 = obstacles_matrix(obs_index,6);

            width_rec2 = obstacles_matrix(obs_index,4);
            height_rec2 = width_rec1;
            depth_rec2 = obstacles_matrix(obs_index,6);
        
            x_rec1 = obstacles_matrix(obs_index,1);
            y_rec1 = obstacles_matrix(obs_index,2);
            z_rec1 = obstacles_matrix(obs_index,3);
        
            x_rec2 = obstacles_matrix(obs_index,1);
            y_rec2 = obstacles_matrix(obs_index,2);
            z_rec2 = obstacles_matrix(obs_index,3);
            
            model.component('comp1').geom('geom1').create(strcat(label,'1'), 'Block');
            model.component('comp1').geom('geom1').feature(strcat(label,'1')).set('pos', [x_rec1 y_rec1 z_rec1]);
            model.component('comp1').geom('geom1').feature(strcat(label,'1')).set('size', [width_rec1 height_rec1 depth_rec1]);
            model.component('comp1').geom('geom1').create(strcat(label,'2'), 'Block');
            model.component('comp1').geom('geom1').feature(strcat(label,'2')).set('pos', [x_rec2 y_rec2 z_rec2]);
            model.component('comp1').geom('geom1').feature(strcat(label,'2')).set('size', [width_rec2 height_rec2 depth_rec2]);

            model.component('comp1').geom('geom1').create(strcat(label,'uni1'), 'Union');
            model.component('comp1').geom('geom1').feature(strcat(label,'uni1')).selection('input').set({strcat(label,'1')...
                strcat(label,'2')});
            labels{obs_index}=strcat(label,'uni1');
            
        end
    end
end
