% Calculates the echo intensity induced by a probe

function[intensity] = calc_intensity(activations)
% INPUTS:
% activations: array of length listlength containing an activation level
%              for each study item.
%
% OUTPUTS:
% intensity: number representing the intensity of the echo.

intensity = 0;

for i = 1:length(activations)
    intensity = intensity + activations(1,i);
end

end