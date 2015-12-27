
%��ȡѵ�����ݣ��������ݺ��������
disp('��ʼ��ȡѵ�����ݣ��������ݺ��������...');

train_x = csvread( 'train_x.csv' );
train_y = csvread( 'train_y.csv' );
test = csvread( 'test_x.csv', 0, 1 );
fid = fopen('features_type.csv');   
dcells = textscan(fid, '%s%s', 'delimiter', ',');
dneeds = dcells(2);   
featureType = vertcat(dneeds{:});
fclose(fid); 

clear fid dcells dneeds;
disp('��ȡѵ�����ݣ��������ݺ�����������...');
disp('��ʼ���ݷ���...');

%��������Ϊ��ֵ�����ͣ����Բ�ͬ���ֿ�����
[r_category, c_category] = find(strcmp(featureType, 'category'));
[r_num, c_num] = find(strcmp(featureType, 'numeric')); 
train_x_category = train_x(:,r_category);
train_x_num = train_x(:,r_num);
test_x_category = test(:,r_category);
test_x_num = test(:,r_num);

clear r_category c_category r_num c_num;
disp('���ݷ������...');
disp('��ʼ������ֵ�͵�ȱʡֵ...');

%�����ݹ�һ������ȱʡֵ����Ϊƽ��ֵ��Ȼ�����������x = (x - mean) / s; sΪ��׼��
% 1. ������ֵ�͵�ȱʡֵ
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
disp('������ֵ�͵�ȱʡֵ����...');
disp('��ʼ������������ݵ�ȱʡֵ,ȫ��ֵΪ0...');

% 2. ������������ݵ�ȱʡֵ,ȫ��ֵΪ0
I = train_x_category < 0;
train_x_category(I) = 0;
I = test_x_category < 0;
test_x_category(I) = 0;

clear I;
disp('������������ݵ�ȱʡֵ,ȫ��ֵΪ0����...');
disp('��ʼ���ݹ�һ��...');

% 3. ���ݹ�һ��
I = std( train_x_num, 0, 1 );
I( I==0 ) = 1;
train_x_num = ( train_x_num - repmat( means', size(train_x_num,1), 1 ) ) ./ (repmat( I, size(train_x_num,1), 1 ));
test_x_num = ( test_x_num - repmat( means', size(test_x_num,1), 1 ) ) ./ (repmat( I, size(test_x_num,1), 1 ));

clear I;
disp('���ݹ�һ������...');
disp('��ʼ����ѵ������...');

%����ѵ������
X = [ train_x_num, train_x_category ];
Y = train_y(:,2);
Test_X = [test_x_num, test_x_category];

clear train_x_num train_x_category train_y test_x_num test_x_category;
disp('����ѵ�����ݽ���...');
disp('��ʼѵ������...');

%ѵ��ģ��
% 1. ��ʼ��theta
init_theta = zeros( 1138, 1 );
% 2. ����finmuc����
options = optimset( 'GradObj', 'on', 'MaxIter', 10 );
% 3. ��ʼѵ��
[theta, cost, exitflag] = fminunc( @(t)(costFunction( t, X, Y, 15000 )), init_theta, options );

disp('ѵ�����ݽ���...');
disp('��ʼԤ������...');

% Ԥ�� 
result = sigmoid( Test_X * theta );

disp('Ԥ�����ݽ���...');
