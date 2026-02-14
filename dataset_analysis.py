import numpy as np
import pandas as pd

def preliminary_descriptive_analysis():
    df = pd.read_csv("social-media-release.csv")
    print(df.head())

    print(f"Columns: {df.columns}")
    # ['id', 'news_headline', 'news_headline_ground_truth', 'post', 'majority_votes', 'class_label']
    print("*"*50)

    example_row = df.iloc[0]
    # print(example_row)

    print(f"ID: {example_row.id}\n",
          f"headline: {example_row.news_headline}\n",
          f"post: {example_row.post}\n",
          f"majority votes: {example_row.majority_votes}\n",
          f"ground truth: {example_row.news_headline_ground_truth}\n",
          f"class label: {example_row.class_label}")

    print("*"*50)

    # get total no of posts
    total_posts = df.shape[0]
    print(f"total posts: {total_posts}")
    print("*"*50)

    no_of_ground_truth = df['news_headline_ground_truth'].value_counts()
    print(f"no ground truth:\n{no_of_ground_truth}")
    ground_truth_perc = (no_of_ground_truth.iloc[0] / total_posts) * 100
    print(f"% ground truth = True: {ground_truth_perc}")
    print("*"*50)

    no_of_class_truth = df['class_label'].value_counts()
    print(f"no class truth:\n{no_of_class_truth}")
    class_truth_perc = (no_of_class_truth.iloc[0] / total_posts) * 100
    print(f"% class truth = True {class_truth_perc}")
    print("*"*50)

    # count no of posts where class label, majority vote and ground truth = true


    # bar chart for each binary t/f column (label, majority, ground truths)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2,2)
           # .figure(figsize=(10,4)))
    colours = ['green', 'red']
    x = ['True', 'False']

    yticks = range(35000, 60000, 5000)

    ax1 = axs[0,0]
    y1 = [no_of_ground_truth.iloc[0], no_of_ground_truth.iloc[1]]
    ax1.bar(x, y1, color=colours)
    ax1.set_title("Ground Truths")
    ax1.set_ylabel("count")
    ax1.set_yticks(yticks)
    ax1.set_ylim(35000, 60000)

    ax2 = axs[1,1]
    y2 = [no_of_class_truth.iloc[0], no_of_class_truth.iloc[1]]
    ax2.bar(x, y2, color=colours)
    ax2.set_title("Class Truth")
    ax2.set_ylabel("count")
    ax2.set_yticks(yticks)
    ax2.set_ylim(35000, 60000)

    plt.tight_layout(pad=2)
    plt.savefig("graphs/t-f_prelim_comparison.png")
    plt.show()



def test():
    print("Test harness")
    preliminary_descriptive_analysis()

if __name__ == "__main__":
    test()