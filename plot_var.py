import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
# You will have to pre-process the data, i.e. calculate all the variance you need.
def plot_training(model_path, epoch):
    df1 = pd.read_csv(os.path.join(model_path, 'progress_1_.csv'))

    df = df1
    df['Epoch'] = df['Epoch'].str.split('/').str[0].astype(int)
    if isinstance(epoch, int):
        df = df[df['Epoch'] <= epoch]
    df['Total Time'] = df['TimeCost(sec)'].cumsum() / 60
    # print(df['Total Time'][182])


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))
    fig.tight_layout(pad=5)  # Add padding between plots

    fig.suptitle("RL Training", fontsize=16, fontweight='bold')

    # Plot 1
    ax1.plot(df['Epoch'], df['Success'], label='Success', color='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Success')

    ax1.fill_between(df['Epoch'],df['s+'], df['s-'], color='blue', alpha=0.2)
    # ax1_ = ax1.twinx()
    # ax1_.bar(df['Epoch'], df['s_v'], color='blue', alpha=0.2)
    # ax1_.axis('off')
    # ax1_.set_ylabel('')

    ax1b = ax1.twinx()
    ax1b.plot(df['Epoch'], df['CollisionsAvg'], label='CollisionsAvg', color='green')
    ax1b.set_ylabel('CollisionsAvg')

    ax1b.fill_between(df['Epoch'], df['c+'], df['c-'], color='green', alpha=0.2)

    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper left',bbox_to_anchor=(0,0.8))

    # ax1b_ = ax1.twinx()
    # ax1b_.bar(df['Epoch'], df['c_v'], color='green', alpha=0.2)
    # ax1b_.axis('off')
    # ax1b_.set_ylabel('CollisionsAvg')

    # Plot 2
    ax2.plot(df['Epoch'], df['ExRewardAvg'], label='ExRewardAvg', color='red')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Reward')

    ax2.fill_between(df['Epoch'], df['e+'], df['e-'], color='red', alpha=0.2)
    ax2.legend(loc='upper left')

    # ax2_ = ax2.twinx()
    # ax2_.bar(df['Epoch'], df['r_v'], color='red', alpha=0.2)
    # ax2_.axis('off')

    # PLot 3
    cl=['CL6','CL5','CL4','CL3','CL2','CL1','']
    cl.reverse()
    x_temp = [1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,
              1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,
              1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,
              1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,
              1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,7,
            ]

    ax3.plot(df['Epoch'], x_temp, label='', color='red',alpha=0)
    ax3.set_yticks(range(len(cl)))
    ax3.set_yticklabels(cl)
    ax3.set_xlabel('Epochs')
    ax3.axhline(y=1, xmin=0.045,xmax=0.65,color='blue', linestyle='-',linewidth=10,alpha=0.4)
    ax3.axhline(y=2, xmin=0.552,xmax=0.672,color='aqua', linestyle='-',linewidth=10,alpha=0.4)
    ax3.axhline(y=3, xmin=0.588,xmax=0.727,color='blue', linestyle='-',linewidth=10,alpha=0.4)
    ax3.axhline(y=4, xmin=0.621, xmax=0.791, color='aqua', linestyle='-', linewidth=10, alpha=0.4)
    ax3.axhline(y=5, xmin=0.696, xmax=0.819, color='blue', linestyle='-', linewidth=10, alpha=0.4)
    ax3.axhline(y=6, xmin=0.723, xmax=0.955, color='aqua', linestyle='-', linewidth=10, alpha=0.4)

    # ax3.set_xticklabels(['1','100','200','300'])
    # Plot Difficulties
#     ax1.axvline(0, color='black', linestyle='--')
#     ax1.text(0, 1.05, "CL1", va='bottom', ha='center')
# # -----------------
#     ax2.axvline(x=0, color='black', linestyle='--')
#     ax2.text(0, -12, "CL1", va='bottom', ha='center')

    # ax2.text(df['Total Time'][99], -12, "CLx", va='bottom', ha='center')

    count = 1
    # for index, row in df.iterrows():
    #     if row['Success'] >= 0.9:
    #         count += 1
    #         ax1.axvline(x=row['Epoch'], color='black', linestyle='--')
    #         ax1.text(row['Epoch'], 1.05, f"CL{count}", va='bottom', ha='center')
    #
    #         ax2.axvline(x=row['Total Time'], color='black', linestyle='--')
    #         ax2.text(row['Total Time'], -12, f"CL{count}", va='bottom', ha='center')
    #
    #     if count == 6:
    #         break

    # save & show
    fig_dir = os.path.join(model_path, 'plots')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    plt.savefig(os.path.join(fig_dir, "training.png"), dpi=300)

if __name__ == "__main__":

    plot_training('log2', 300)
    plt.show()