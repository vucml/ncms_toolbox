function record = record_fields(record, net, field)
%
%
%

% determine data kind
% scalar
% vector
% 2-d matrix

for fs = 1:length(field)
  
  if isfield(net, field{fs})
    
    % determine data kind
    data_size = size(net.(field{fs}));
    % scalar
    if isequal(data_size, [1 1])
      record.(field{fs}).data(end+1) = net.(field{fs});
    end
    % vector
    if all([length(data_size)==2,any(data_size==1),any(data_size>1)])
      record.(field{fs}).data(:,end+1) = net.(field{fs});
    end
    % 2d matrix
    if all([length(data_size)==2,all(data_size>1)])
      record.(field{fs}).data(:,:,end+1) = net.(field{fs});
    end
  
  end % isfield
  
end % for fields


