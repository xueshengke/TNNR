function obj = nuclear_norm(X)
    if size(X, 1) > 2000
        sigma = svds(X, 100);
    else
        sigma = svd(X);
    end
    obj = sum(sigma);
end