"""
Dataset Analysis Script for Vietnamese Fake News Detection
Generates visualizations and statistics for the report
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import argparse
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure matplotlib for better font support
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def load_dataset(train_path, test_path=None):
    """Load and combine datasets"""
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    train_df['split'] = 'train'
    
    if test_path and os.path.exists(test_path):
        print(f"Loading test data from: {test_path}")
        test_df = pd.read_csv(test_path)
        test_df['split'] = 'test'
        # Only combine if test has labels
        if 'label' in test_df.columns:
            df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            df = train_df
    else:
        df = train_df
    
    return df


def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters but keep Vietnamese
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    return text.strip()


def analyze_label_distribution(df, output_dir):
    """Analyze and plot label distribution"""
    print("\n=== Label Distribution ===")
    
    label_counts = df['label'].value_counts().sort_index()
    print(f"Real News (0): {label_counts.get(0, 0)}")
    print(f"Fake News (1): {label_counts.get(1, 0)}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    colors = ['#2ecc71', '#e74c3c']  # Green for real, red for fake
    bars = axes[0].bar(['Real News', 'Fake News'], 
                       [label_counts.get(0, 0), label_counts.get(1, 0)],
                       color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_xlabel('Label', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Label Distribution (Count)', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars, [label_counts.get(0, 0), label_counts.get(1, 0)]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    sizes = [label_counts.get(0, 0), label_counts.get(1, 0)]
    labels = ['Real News', 'Fake News']
    explode = (0, 0.05)
    axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90,
                textprops={'fontsize': 11})
    axes[1].set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'label_distribution.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")
    
    return label_counts


def analyze_text_length(df, output_dir):
    """Analyze text length distribution"""
    print("\n=== Text Length Analysis ===")
    
    df['text_length'] = df['post_message'].astype(str).apply(len)
    df['word_count'] = df['post_message'].astype(str).apply(lambda x: len(x.split()))
    
    # Statistics by label
    for label in [0, 1]:
        subset = df[df['label'] == label]
        label_name = "Real News" if label == 0 else "Fake News"
        print(f"\n{label_name}:")
        print(f"  Avg character length: {subset['text_length'].mean():.1f}")
        print(f"  Avg word count: {subset['word_count'].mean():.1f}")
        print(f"  Min/Max characters: {subset['text_length'].min()} / {subset['text_length'].max()}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Character length distribution by label
    for label, color, name in [(0, '#2ecc71', 'Real News'), (1, '#e74c3c', 'Fake News')]:
        subset = df[df['label'] == label]['text_length']
        axes[0, 0].hist(subset, bins=50, alpha=0.6, label=name, color=color, edgecolor='black')
    axes[0, 0].set_xlabel('Character Length', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Character Length Distribution by Label', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, df['text_length'].quantile(0.95))
    
    # Word count distribution by label
    for label, color, name in [(0, '#2ecc71', 'Real News'), (1, '#e74c3c', 'Fake News')]:
        subset = df[df['label'] == label]['word_count']
        axes[0, 1].hist(subset, bins=50, alpha=0.6, label=name, color=color, edgecolor='black')
    axes[0, 1].set_xlabel('Word Count', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Word Count Distribution by Label', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, df['word_count'].quantile(0.95))
    
    # Box plot - Character length
    df_melted = df[['label', 'text_length']].copy()
    df_melted['label'] = df_melted['label'].map({0: 'Real News', 1: 'Fake News'})
    sns.boxplot(data=df_melted, x='label', y='text_length', ax=axes[1, 0],
                palette={'Real News': '#2ecc71', 'Fake News': '#e74c3c'})
    axes[1, 0].set_xlabel('Label', fontsize=11)
    axes[1, 0].set_ylabel('Character Length', fontsize=11)
    axes[1, 0].set_title('Character Length by Label (Box Plot)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0, df['text_length'].quantile(0.95))
    
    # Box plot - Word count
    df_melted = df[['label', 'word_count']].copy()
    df_melted['label'] = df_melted['label'].map({0: 'Real News', 1: 'Fake News'})
    sns.boxplot(data=df_melted, x='label', y='word_count', ax=axes[1, 1],
                palette={'Real News': '#2ecc71', 'Fake News': '#e74c3c'})
    axes[1, 1].set_xlabel('Label', fontsize=11)
    axes[1, 1].set_ylabel('Word Count', fontsize=11)
    axes[1, 1].set_title('Word Count by Label (Box Plot)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim(0, df['word_count'].quantile(0.95))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'text_length_analysis.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")
    
    return df


def analyze_engagement_metrics(df, output_dir):
    """Analyze engagement metrics (likes, shares, comments)"""
    print("\n=== Engagement Metrics Analysis ===")
    
    # Convert to numeric, handling 'unknown' values
    for col in ['num_like_post', 'num_share_post', 'num_comment_post']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check if engagement columns exist and have valid data
    engagement_cols = ['num_like_post', 'num_share_post', 'num_comment_post']
    available_cols = [col for col in engagement_cols if col in df.columns and df[col].notna().sum() > 0]
    
    if not available_cols:
        print("No engagement metrics available")
        return df
    
    fig, axes = plt.subplots(1, len(available_cols), figsize=(5*len(available_cols), 5))
    if len(available_cols) == 1:
        axes = [axes]
    
    col_names = {
        'num_like_post': 'Likes',
        'num_share_post': 'Shares',
        'num_comment_post': 'Comments'
    }
    
    for idx, col in enumerate(available_cols):
        for label, color, name in [(0, '#2ecc71', 'Real'), (1, '#e74c3c', 'Fake')]:
            subset = df[(df['label'] == label) & df[col].notna()][col]
            if len(subset) > 0:
                # Use log scale for better visualization
                subset_log = np.log1p(subset)
                axes[idx].hist(subset_log, bins=30, alpha=0.6, label=name, color=color, edgecolor='black')
        
        axes[idx].set_xlabel(f'Log({col_names[col]} + 1)', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'{col_names[col]} Distribution', fontsize=12, fontweight='bold')
        axes[idx].legend()
        
        # Print statistics
        for label in [0, 1]:
            subset = df[(df['label'] == label) & df[col].notna()][col]
            label_name = "Real" if label == 0 else "Fake"
            print(f"  {label_name} - {col_names[col]}: mean={subset.mean():.1f}, median={subset.median():.1f}")
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'engagement_metrics.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")
    
    return df


def analyze_word_frequency(df, output_dir, top_n=20):
    """Analyze most common words"""
    print("\n=== Word Frequency Analysis ===")
    
    # Vietnamese stop words (basic list)
    stop_words = set(['và', 'của', 'là', 'có', 'được', 'cho', 'các', 'với', 'này', 
                      'trong', 'để', 'không', 'một', 'những', 'đã', 'từ', 'như', 
                      'về', 'ra', 'khi', 'người', 'cũng', 'nhưng', 'vào', 'nếu',
                      'hay', 'sẽ', 'đến', 'rất', 'nên', 'lại', 'còn', 'theo',
                      'thì', 'tại', 'hơn', 'đây', 'đó', 'bị', 'mà', 'ở'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (label, title, color) in enumerate([(0, 'Real News', '#2ecc71'), (1, 'Fake News', '#e74c3c')]):
        subset = df[df['label'] == label]['post_message'].astype(str)
        
        # Tokenize and count words
        all_words = []
        for text in subset:
            cleaned = clean_text(text).lower()
            words = [w for w in cleaned.split() if len(w) > 1 and w not in stop_words and not w.isdigit()]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_n)
        
        if top_words:
            words, counts = zip(*top_words)
            y_pos = np.arange(len(words))
            
            axes[idx].barh(y_pos, counts, color=color, edgecolor='black')
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(words)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Frequency', fontsize=11)
            axes[idx].set_title(f'Top {top_n} Words - {title}', fontsize=12, fontweight='bold')
            
            # Add count labels
            for i, count in enumerate(counts):
                axes[idx].text(count + max(counts)*0.01, i, f'{count:,}', 
                              va='center', fontsize=9)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'word_frequency.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def generate_summary_statistics(df, output_dir):
    """Generate summary statistics table"""
    print("\n=== Summary Statistics ===")
    
    stats = {
        'Metric': [],
        'Overall': [],
        'Real News': [],
        'Fake News': []
    }
    
    # Total samples
    stats['Metric'].append('Total Samples')
    stats['Overall'].append(len(df))
    stats['Real News'].append(len(df[df['label'] == 0]))
    stats['Fake News'].append(len(df[df['label'] == 1]))
    
    # Average text length
    stats['Metric'].append('Avg. Character Length')
    stats['Overall'].append(f"{df['post_message'].astype(str).apply(len).mean():.1f}")
    stats['Real News'].append(f"{df[df['label']==0]['post_message'].astype(str).apply(len).mean():.1f}")
    stats['Fake News'].append(f"{df[df['label']==1]['post_message'].astype(str).apply(len).mean():.1f}")
    
    # Average word count
    stats['Metric'].append('Avg. Word Count')
    stats['Overall'].append(f"{df['post_message'].astype(str).apply(lambda x: len(x.split())).mean():.1f}")
    stats['Real News'].append(f"{df[df['label']==0]['post_message'].astype(str).apply(lambda x: len(x.split())).mean():.1f}")
    stats['Fake News'].append(f"{df[df['label']==1]['post_message'].astype(str).apply(lambda x: len(x.split())).mean():.1f}")
    
    # Engagement metrics if available
    for col, name in [('num_like_post', 'Avg. Likes'), ('num_share_post', 'Avg. Shares'), ('num_comment_post', 'Avg. Comments')]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            stats['Metric'].append(name)
            stats['Overall'].append(f"{df[col].mean():.1f}")
            stats['Real News'].append(f"{df[df['label']==0][col].mean():.1f}")
            stats['Fake News'].append(f"{df[df['label']==1][col].mean():.1f}")
    
    stats_df = pd.DataFrame(stats)
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(10, len(stats['Metric']) * 0.6 + 1))
    ax.axis('off')
    
    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db', '#95a5a6', '#2ecc71', '#e74c3c']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Vietnamese Fake News Dataset Summary', fontsize=14, fontweight='bold', pad=20)
    
    filepath = os.path.join(output_dir, 'summary_statistics.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")
    
    # Also save as CSV
    csv_path = os.path.join(output_dir, 'summary_statistics.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    return stats_df


def create_combined_overview(df, output_dir):
    """Create a combined overview figure"""
    print("\n=== Creating Combined Overview ===")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Label distribution pie chart (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    label_counts = df['label'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie([label_counts.get(0, 0), label_counts.get(1, 0)], 
            labels=['Real', 'Fake'], colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Label Distribution', fontsize=12, fontweight='bold')
    
    # 2. Label count bar chart (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    bars = ax2.bar(['Real News', 'Fake News'], 
                   [label_counts.get(0, 0), label_counts.get(1, 0)],
                   color=colors, edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_title('Sample Count by Label', fontsize=12, fontweight='bold')
    for bar, count in zip(bars, [label_counts.get(0, 0), label_counts.get(1, 0)]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    # 3. Text length distribution (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    df['text_length'] = df['post_message'].astype(str).apply(len)
    for label, color in [(0, '#2ecc71'), (1, '#e74c3c')]:
        subset = df[df['label'] == label]['text_length']
        ax3.hist(subset, bins=40, alpha=0.6, color=color, 
                label='Real' if label == 0 else 'Fake', edgecolor='black')
    ax3.set_xlabel('Character Length')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Text Length Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.set_xlim(0, df['text_length'].quantile(0.95))
    
    # 4. Word count box plot (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    df['word_count'] = df['post_message'].astype(str).apply(lambda x: len(x.split()))
    df_box = df[['label', 'word_count']].copy()
    df_box['label'] = df_box['label'].map({0: 'Real', 1: 'Fake'})
    sns.boxplot(data=df_box, x='label', y='word_count', ax=ax4,
                palette={'Real': '#2ecc71', 'Fake': '#e74c3c'})
    ax4.set_xlabel('Label')
    ax4.set_ylabel('Word Count')
    ax4.set_title('Word Count by Label', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, df['word_count'].quantile(0.95))
    
    # 5. Character length box plot (bottom middle)
    ax5 = fig.add_subplot(2, 3, 5)
    df_box = df[['label', 'text_length']].copy()
    df_box['label'] = df_box['label'].map({0: 'Real', 1: 'Fake'})
    sns.boxplot(data=df_box, x='label', y='text_length', ax=ax5,
                palette={'Real': '#2ecc71', 'Fake': '#e74c3c'})
    ax5.set_xlabel('Label')
    ax5.set_ylabel('Character Length')
    ax5.set_title('Character Length by Label', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, df['text_length'].quantile(0.95))
    
    # 6. Statistics text (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    Dataset Statistics
    ──────────────────────
    Total Samples: {len(df):,}
    
    Real News: {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)
    Fake News: {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)
    
    Avg. Character Length:
      • Real: {df[df['label']==0]['text_length'].mean():.0f}
      • Fake: {df[df['label']==1]['text_length'].mean():.0f}
    
    Avg. Word Count:
      • Real: {df[df['label']==0]['word_count'].mean():.0f}
      • Fake: {df[df['label']==1]['word_count'].mean():.0f}
    """
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.suptitle('Vietnamese Fake News Dataset Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'dataset_overview.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Vietnamese Fake News Dataset')
    parser.add_argument('--train_path', type=str, 
                        default='../vn_fake_news/Data/train.csv',
                        help='Path to training CSV')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test CSV (optional)')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    df = load_dataset(args.train_path, args.test_path)
    
    # Filter to only labeled data
    if 'label' in df.columns:
        df = df.dropna(subset=['label', 'post_message'])
        df['label'] = df['label'].astype(int)
    else:
        print("Error: No 'label' column found in dataset")
        return
    
    print(f"\nTotal samples: {len(df)}")
    
    # Run analyses
    analyze_label_distribution(df, args.output_dir)
    analyze_text_length(df, args.output_dir)
    analyze_engagement_metrics(df, args.output_dir)
    analyze_word_frequency(df, args.output_dir)
    generate_summary_statistics(df, args.output_dir)
    create_combined_overview(df, args.output_dir)
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print(f"All figures saved to: {args.output_dir}")
    print("="*50)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        filepath = os.path.join(args.output_dir, f)
        size = os.path.getsize(filepath) / 1024
        print(f"  - {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()
