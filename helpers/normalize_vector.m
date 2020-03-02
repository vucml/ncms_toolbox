function norm_vec = normalize_vector(vec)
%NORMALIZE_VECTOR   Normalize a vector to length 1.
%
%  norm_vec = normalize_vector(vec);

if ~isvector(vec)
  error('Function requires vector input.');
end

if norm(vec) ~= 0
  norm_vec = vec / norm(vec);
else
  norm_vec = vec;
end

