function [jVal, gradient] = costFunction( theta, X, Y, m )

    z = theta' * (X');
    h = sigmoid( z );  %[1, m]
    I = ones( m, 1 );

    jVal = ( ( -Y .* log(h') ) - ( I - Y ) .* log( I - h' ) )' * I;

    lambda = 10;  %Using for regularized regression, just for testing.
    jVal = jVal / m;

    g = sum ( X .* repmat( (h'-Y), 1, size(X,2) ), 1 ); 
    gradient = g' ./ m;

end