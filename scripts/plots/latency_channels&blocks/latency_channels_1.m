% data
channels = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
latencies = [53, 69, 70, 67, 68, 94, 94, 90, 86, 126, 125, 104, 73, 145, 103, 158, 103, 171, 119, 185, 89];

% plot
plot(channels, latencies, '-x', 'LineWidth', 2, 'MarkerSize', 10)
xlim([4, 24]);
set(gca, 'FontSize',20, 'Fontname', 'Times New Roman');
set(gca,'xtick', 4:4:24);
xlabel('Number of channels')
ylabel('Latency (ms)')
