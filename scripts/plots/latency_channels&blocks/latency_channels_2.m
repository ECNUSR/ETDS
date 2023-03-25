% data
channels = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64];
latencies = [53, 68, 86, 73, 103, 89, 97, 100, 130, 136, 148, 150, 178, 183, 201, 190];

% plot
plot(channels, latencies, '-x', 'LineWidth', 2, 'MarkerSize', 10)
xlim([4, 64]);
set(gca, 'FontSize',20, 'Fontname', 'Times New Roman');
set(gca,'xtick', 4:12:64);
xlabel('Number of channels')
ylabel('Latency (ms)')
