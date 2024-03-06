import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()

def plot_training(model_path, epoch):
    df = pd.read_csv(os.path.join(model_path, 'progress.csv'))

    df['Epoch'] = df['Epoch'].str.split('/').str[0].astype(int)
    if isinstance(epoch, int):
        df = df[df['Epoch'] <= epoch]
    df['Total Time'] = df['TimeCost(sec)'].cumsum() / 60

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    fig.tight_layout(pad=5)  # Add padding between plots

    fig.suptitle("RL Training", fontsize=16, fontweight='bold')

    # Plot 1
    ax1.plot(df['Epoch'], df['Success'], label='Success', color='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Success')

    ax1b = ax1.twinx()
    ax1b.plot(df['Epoch'], df['CollisionsAvg'], label='CollisionsAvg', color='green')
    ax1b.set_ylabel('CollisionsAvg')

    # Plot 2
    ax2.plot(df['Total Time'], df['ExRewardAvg'], label='ExRewardAvg', color='red')
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Reward')

    # Plot Difficulties
    ax1.axvline(0, color='black', linestyle='--')
    ax1.text(0, 1.05, "C1", va='bottom', ha='center')

    ax2.axvline(x=0, color='black', linestyle='--')
    ax2.text(0, -19, "C1", va='bottom', ha='center')
    count = 1
    for index, row in df.iterrows():
        if row['Success'] >= 0.9:
            count += 1
            ax1.axvline(x=row['Epoch'], color='black', linestyle='--')
            ax1.text(row['Epoch'], 1.05, f"C{count}", va='bottom', ha='center')

            ax2.axvline(x=row['Total Time'], color='black', linestyle='--')
            ax2.text(row['Total Time'], -19, f"C{count}", va='bottom', ha='center')

        if count == 6:
            break

    # save & show
    fig_dir = os.path.join(model_path, 'plots')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    plt.savefig(os.path.join(fig_dir, "training.png"), dpi=300)


