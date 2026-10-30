
% K = sparse(rows + 1, cols + 1, values);
% spy(K);
function Vv = nullspace(name, data_dir, mode_count)
if nargin < 2, data_dir = fullfile('data', 'eigs'); end
if nargin < 3, mode_count = 30; end
% load data/eigs/squishy_ball_lowlow.mat
% load data/eigs/boatv5.mat
load(fullfile(data_dir, [name '.mat']));
% load data/eigs/tmp.mat
% load output/bug/A1.mat;
% na1 = null(A1');

% tilde_K = na1' * K * na1;
% tilde_M = na1' * M * na1;
tilde_K = K;
tilde_M = M;

[Vv, D] = eigs(tilde_K, tilde_M, min(mode_count, size(K, 1) - 1), 'smallestabs');

% diagVMV = diag(Vv' * tilde_M * Vv);
% Vv = Vv ./ sqrt(diagVMV');
% V = na1 * Vv;
% save data/eigs/Q_squishy_ball_lowlow.mat Vv D
% save data/eigs/Q_boatv5.mat Vv D
save(fullfile(data_dir, ['Q_' name '.mat']), 'Vv', 'D');

end
