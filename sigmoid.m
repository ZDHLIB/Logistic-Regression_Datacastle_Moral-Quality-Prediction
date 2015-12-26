function h = sigmoid(z)

    h = zeros( size(z) );

    for i = 1 : size( z, 1 ) 
      for j = 1 : size( z, 2 )
        h(i, j) = 1 ./ ( 1 + exp(1)^(-z(i, j) ) );
      end;
    end;
end