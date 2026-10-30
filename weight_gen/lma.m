function Vv = lma(name, data_dir, mode_count)
if nargin < 2, data_dir = fullfile('data', 'eigs_lma'); end
if nargin < 3, mode_count = 30; end
load(fullfile(data_dir, [name '.mat']));
tilde_K = K;
tilde_M = M;

[Vv, D] = eigs(tilde_K, tilde_M, min(mode_count, size(K, 1) - 1), 'smallestabs');
save(fullfile(data_dir, ['Q_' name '.mat']), 'Vv', 'D');

end
