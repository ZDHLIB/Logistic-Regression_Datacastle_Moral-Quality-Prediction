
%读取训练数据，测试数据和数据类别
disp('开始读取训练数据，测试数据和数据类别...');

train_x = csvread( 'train_x.csv' );
train_y = csvread( 'train_y.csv' );
test = csvread( 'test_x.csv', 0, 1 );
fid = fopen('features_type.csv');   
dcells = textscan(fid, '%s%s', 'delimiter', ',');
dneeds = dcells(2);   
featureType = vertcat(dneeds{:});
fclose(fid); 

clear fid dcells dneeds;
disp('读取训练数据，测试数据和数据类别结束...');
disp('开始数据分离...');

%数据类别分为数值与类型，所以不同类别分开处理
[r_category, c_category] = find(strcmp(featureType, 'category'));
[r_num, c_num] = find(strcmp(featureType, 'numeric')); 
train_x_category = train_x(:,r_category);
train_x_num = train_x(:,r_num);
test_x_category = test(:,r_category);
test_x_num = test(:,r_num);

clear r_category c_category r_num c_num;
disp('数据分离结束...');
disp('开始处理数值型的缺省值...');

%对数据归一化处理，缺省值设置为平均值，然后对所有数据x = (x - mean) / s; s为标准差
% 1. 处理数值型的缺省值
means = zeros(size(train_x_num, 2),1);
for i = 1 : size(train_x_num, 2)
    t = train_x_num(:,i);
    I = t >= 0 ;
    means(i) = mean(t(I),1);
    
    I =  t < 0 ;
    t(I) = means(i);
    train_x_num( :, i ) = t;
    
    I = test_x_num(:,i) < 0;
    test_x_num(I) = means(i);
end;

clear i t I;
disp('处理数值型的缺省值结束...');
disp('开始处理类别型数据的缺省值,全赋值为0...');

% 2. 处理类别型数据的缺省值,全赋值为0
I = train_x_category < 0;
train_x_category(I) = 0;
I = test_x_category < 0;
test_x_category(I) = 0;

clear I;
disp('处理类别型数据的缺省值,全赋值为0结束...');
disp('开始数据归一化...');

% 3. 数据归一化
I = std( train_x_num, 0, 1 );
I( I==0 ) = 1;
train_x_num = ( train_x_num - repmat( means', size(train_x_num,1), 1 ) ) ./ (repmat( I, size(train_x_num,1), 1 ));
test_x_num = ( test_x_num - repmat( means', size(test_x_num,1), 1 ) ) ./ (repmat( I, size(test_x_num,1), 1 ));

clear I;
disp('数据归一化结束...');
disp('开始整合训练数据...');

%整合训练数据
X = [ train_x_num, train_x_category ];
Y = train_y(:,2);
Test_X = [test_x_num, test_x_category];

clear train_x_num train_x_category train_y test_x_num test_x_category;
disp('整合训练数据结束...');
disp('开始训练数据...');

%训练模型
% 1. 初始化theta
init_theta = zeros( 1138, 1 );
% 2. 配置finmuc参数
options = optimset( 'GradObj', 'on', 'MaxIter', 10 );
% 3. 开始训练
[theta, cost, exitflag] = fminunc( @(t)(costFunction( t, X, Y, 15000 )), init_theta, options );

disp('训练数据结束...');
disp('开始预测数据...');

% 预测 
result = sigmoid( Test_X * theta );

disp('预测数据结束...');
