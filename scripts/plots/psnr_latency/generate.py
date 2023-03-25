''' Trade-off diagram of psnr and latency. '''
import matplotlib.pyplot as plt


def main():
    ''' main '''
    # hyp
    markersize = 15
    fontsize = 30
    text_fontsize = 20

    fig, ax = plt.subplots(figsize=(15, 10))

    # ESPCN
    ax.plot(9.79, 32.8479, marker='o', linestyle='dashed', markersize=markersize, color='#1F77B4')
    plt.annotate('ESPCN', (9.79 + 0.2, 32.8479 + 0.03), fontsize=text_fontsize, color='#1F77B4')

    # FSRCNN
    ax.plot(13.7, 33.0111, marker='v', linestyle='dashed', markersize=markersize, color='#FF7F0E')
    plt.annotate('FSRCNN', (13.7 - 0.2, 33.0111 - 0.07), fontsize=text_fontsize, color='#FF7F0E')

    # ECBSR
    ax.plot([16.2, 19.7, 28.1], [33.0693, 33.5625, 33.6737], marker='s', linestyle='dashed', markersize=markersize, color='#2CA02C')
    plt.annotate('ECBSR', (19.7 - 1.2, 33.5625 - 0.25), fontsize=text_fontsize, color='#2CA02C', rotation=62)

    # ECBSR+ET
    ax.plot([9.35, 13.1, 21.5], [33.0693, 33.5625, 33.6737], marker='s', linestyle='dashed', markersize=markersize, color='#E377C2')
    plt.annotate('ECBSR+ET (Ours)', (13.5 - 2.8, 33.5625 - 0.43), fontsize=text_fontsize, color='#E377C2', rotation=58)

    # ABPN
    ax.plot([20.5, 29.2], [33.5293, 33.7054], marker='H', linestyle='dashed', markersize=markersize, color='#9467BD')
    plt.annotate('ABPN', (29.2 - 5, 33.7054 - 0.15), fontsize=text_fontsize, color='#9467BD', rotation=20)

    # ABPN+ET
    ax.plot([13.6, 22.2], [33.5293, 33.7054], marker='H', linestyle='dashed', markersize=markersize, color='#8C564B')
    plt.annotate('ABPN+ET (Ours)', (22.3 - 7.5, 33.7054 - 0.19), fontsize=text_fontsize, color='#8C564B', rotation=14)

    # ETDS
    ax.plot([8.3, 12.1, 13.9, 25.2], [33.1382, 33.4866, 33.6457, 33.8768], marker='*', linestyle='dashed', markersize=markersize, color='#D62728')
    plt.annotate('ETDS (Ours)', (14.7 + 2.8, 33.6457 + 0.1), fontsize=text_fontsize, color='#D62728', rotation=15)

    plt.xlabel('Latency (ms)', fontsize=fontsize)
    plt.ylabel('PSNR (dB)', fontsize=fontsize)

    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize(f'{fontsize}')
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize(f'{fontsize}')


    ax.grid(b=True, linestyle='-.', linewidth=0.5)

    plt.show()

    fig.savefig('scripts/plots/psnr_latency/psnr_latency.pdf', format='pdf', dpi=300)


if __name__ == '__main__':
    main()
