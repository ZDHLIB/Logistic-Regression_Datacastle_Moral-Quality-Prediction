%


%��ȡѵ�����ݣ��������ݺ��������
test = csvread( 'test_x.csv', 1, 1 );
train_x = csvread( 'train_x.csv', 1, 1 );
train_y = csvread( 'train_y.csv', 1, 1 );
fid = fopen('features_type.csv');  
fclose(fid);  
dcells = textscan(fid, '%s%s', 'delimiter', ',');
dneeds = dcells(2);   
featureType = vertcat(dneeds{:});

%��������Ϊ��ֵ�����ͣ����Բ�ͬ���ֿ�����
[r_category, c_category] = find(strcmp(featureType, '"category"'));
[r_num, c_num] = find(strcmp(featureType, '"numeric"')); 
train_x_category = train_x(:,r_category);
train_x_num = train_x(:,r_num);
test_x_category = test(:,r_category);
test_x_num = test(:,r_num);

%�����ݹ�һ������ȱʡֵ����Ϊƽ��ֵ��Ȼ�����������x = (x - mean) / s; sΪ��׼��
% 1. ������ֵ�͵�ȱʡֵ
means = zeros(size(train_x_num, 1),1);
for i = 1 : size(train_x_num, 1)
    t = train_x_num(:,i);
    I = t >= 0 ;
    means(i) = mean(t(I),1);
    
    I =  t < 0 ;
    t(I) = means(i);
    train_x_num( :, i ) = t;
    
    I = test_x_num(:,i) < 0;
    test_x_num(I) = means(i);
end;

% 2. ������������ݵ�ȱʡֵ,ȫ��ֵΪ0
I = train_x_category < 0;
train_x_category(I) = 0;
I = test_x_category < 0;
test_x_category(I) = 0;

% 3. ���ݹ�һ��
I = std( train_x_num, 0, 1 );
I( I==0 ) = 1;
train_x_num = ( train_x_num - repmat( means', size(train_x_num,1), 1 ) ) ./ I;

%����ѵ������
X = [ train_x_num, train_x_category ];
Y = train_y;
Test_X = [test_x_num, test_x_category];

%ѵ��ģ��
% 1. ��ʼ��theta
init_theta = zeros( n, 1 );
% 2. ����finmuc����
options = optimset( 'GradObj', 'on', 'MaxIter', 500 );
% 3. ��ʼѵ��
[theta, cost] = fminunc( @(t)(costFunction( t, X, Y, m )), init_theta, options );

% Ԥ�� 
result = zeros( size(Test_X, 2), 1 );
for i = 1 : size(Test_X, 2)
  result(i) = sigmoid( Test_X( i, : ) * theta );
end;




