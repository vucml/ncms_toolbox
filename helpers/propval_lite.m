function [merged, unused] = propval_lite(propvals, defaults)
%PROPVAL_LITE   Like propval, but simpler and faster.
%
%  [merged, unused] = propval_lite(propvals, defaults)

if iscell(propvals) && isscalar(propvals)
  propvals = propvals{1};
end
  
merged = defaults;
unused = {};

if isstruct(propvals)
  props = fieldnames(propvals);
  vals = struct2cell(propvals);
else
  props = propvals(1:2:end);
  vals = propvals(2:2:end);
end

for i = 1:length(props)
  if nargout < 2 || isfield(merged, props{i})
    merged.(props{i}) = vals{i};
  else
    unused{end+1} = props{i};
    unused{end+1} = vals{i};
  end
end
