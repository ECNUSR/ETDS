% data
leyers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
latencies = [59, 63.5, 68.3, 73, 78, 83, 88, 93, 98, 103];

% plot
plot(leyers, latencies, '-x', 'LineWidth', 2, 'MarkerSize', 10)
xlim([1, 10]);
set(gca, 'FontSize',20, 'Fontname', 'Times New Roman');
xlabel('Number of layers')
ylabel('Latency (ms)')
