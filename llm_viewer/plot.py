import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
# Hardware = 'nvidia_H100'
# Hardware = 'nvidia_A100_80G'
Hardware="nvidia_H100_PCIe"
# Hardware="nvidia_A100"

GQA_path = f'Llama3.2-1B_results_{Hardware}.csv'
QLA_path = f'Llama3.2-gla-1B_results_{Hardware}.csv'

# Function to extract data from the raw text with better time unit handling
def extract_data(content, framework_name):
    data = []
    
    # Pattern to extract batch size and sequence length from section headers
    header_pattern = r"=== .*?batchsize=(\d+) seqlen=(\d+) tp_size=\d+ ==="
    
    # Pattern to extract inference time with unit
    decode_pattern = r"decode,.*?,(\d+\.\d+)(us|ms|s)"
    prefill_pattern = r"prefill,.*?,(\d+\.\d+)(us|ms|s)"
    
    lines = content.strip().split('\n')
    
    batch_size = None
    seq_len = None
    
    for line in lines:
        # Check if this is a header line
        header_match = re.search(header_pattern, line)
        if header_match:
            batch_size = int(header_match.group(1))
            seq_len = int(header_match.group(2))
            continue
        
        # Check if this is a decode line
        decode_match = re.search(decode_pattern, line)
        if decode_match and batch_size is not None and seq_len is not None:
            decode_time = float(decode_match.group(1))
            time_unit = decode_match.group(2)
            
            # Convert to ms for consistency
            if time_unit == "us":
                decode_time_ms = decode_time / 1000  # us to ms
            elif time_unit == "s":
                decode_time_ms = decode_time * 1000  # s to ms
            else:
                decode_time_ms = decode_time  # already in ms
            
            data.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'operation': 'decode',
                'inference_time_ms': decode_time_ms,
                'framework': framework_name
            })
            continue
        
        # Check if this is a prefill line
        prefill_match = re.search(prefill_pattern, line)
        if prefill_match and batch_size is not None and seq_len is not None:
            prefill_time = float(prefill_match.group(1))
            time_unit = prefill_match.group(2)
            
            # Convert to ms if needed
            if time_unit == "us":
                prefill_time_ms = prefill_time / 1000  # us to ms
            elif time_unit == "s":
                prefill_time_ms = prefill_time * 1000  # s to ms
            else:
                prefill_time_ms = prefill_time  # already in ms
            
            data.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'operation': 'prefill',
                'inference_time_ms': prefill_time_ms,
                'framework': framework_name
            })
    
    return pd.DataFrame(data)

# Extract data from both files
try:
    with open(GQA_path, 'r') as f:
        content1 = f.read()
    
    with open(QLA_path, 'r') as f:
        content2 = f.read()
    
    df1 = extract_data(content1, 'GQA')
    df2 = extract_data(content2, 'GLA')
    
    # Combine the dataframes
    df = pd.concat([df1, df2])
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure both CSV files are in the current directory.")
    exit(1)

# Set style
sns.set(style="whitegrid")
sns.set_palette("colorblind")

# 1. Line plots comparing both frameworks across sequence lengths for all batch sizes
batch_sizes = sorted(df['batch_size'].unique())

plt.figure(figsize=(20, 10))

# Decode operation
palette = sns.color_palette()
plt.subplot(1, 2, 1)
for i, bs in enumerate(batch_sizes):
    for framework in ['GQA', 'GLA']:
        subset = df[(df['batch_size'] == bs) & (df['operation'] == 'decode') & (df['framework'] == framework)]
        if not subset.empty:
            plt.plot(subset['seq_len'], subset['inference_time_ms'], 
                     marker='o' if framework == 'GQA' else 's',
                     linestyle='-' if framework == 'GQA' else '--',
                     color=palette[i],
                     label=f'{framework}, Batch={bs}')

plt.title('Decode Time Comparison', fontsize=14)
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Inference Time (ms)', fontsize=12)
plt.grid(True)
plt.legend()
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(sorted(df['seq_len'].unique()))  # Ensure all seq lengths are displayed

# Prefill operation - split into multiple plots for better visualization
palette = sns.color_palette()
plt.subplot(1, 2, 2)
for i, bs in enumerate(batch_sizes):
    for framework in ['GQA', 'GLA']:
        subset = df[(df['batch_size'] == bs) & (df['operation'] == 'prefill') & (df['framework'] == framework)]
        if not subset.empty:
            plt.plot(subset['seq_len'], subset['inference_time_ms'], 
                     marker='o' if framework == 'GQA' else 's',
                     linestyle='-' if framework == 'GQA' else '--',
                     color=palette[i],
                     label=f'{framework}, Batch={bs}')

plt.title('Prefill Time Comparison', fontsize=14)
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Inference Time (ms)', fontsize=12)
plt.grid(True)
plt.legend()
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(sorted(df['seq_len'].unique()))  # Ensure all seq lengths are displayed

plt.tight_layout()
plt.savefig(f'{Hardware}_framework_comparison_by_sequence_all.png', dpi=300)

# 2. Create separate plots for smaller batch sizes to see details better
small_batches = [1, 2, 4]
medium_batches = [8, 16]
large_batches = [32]

# Create a function to generate plots for specific batch size ranges
def plot_batch_range(batch_range, title_suffix):
    plt.figure(figsize=(20, 10))
    
    # Decode operation
    plt.subplot(1, 2, 1)
    for bs in batch_range:
        for framework in ['GQA', 'GLA']:
            subset = df[(df['batch_size'] == bs) & (df['operation'] == 'decode') & (df['framework'] == framework)]
            if not subset.empty:
                plt.plot(subset['seq_len'], subset['inference_time_ms'], 
                         marker='o' if framework == 'GQA' else 's',
                         linestyle='-' if framework == 'GQA' else '--',
                         label=f'{framework}, Batch={bs}')
    
    plt.title(f'Decode Time Comparison - {title_suffix}', fontsize=14)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(sorted(df['seq_len'].unique()))  # Ensure all seq lengths are displayed
    
    # Prefill operation
    plt.subplot(1, 2, 2)
    for bs in batch_range:
        for framework in ['GQA', 'GLA']:
            subset = df[(df['batch_size'] == bs) & (df['operation'] == 'prefill') & (df['framework'] == framework)]
            if not subset.empty:
                plt.plot(subset['seq_len'], subset['inference_time_ms'], 
                         marker='o' if framework == 'GQA' else 's',
                         linestyle='-' if framework == 'GQA' else '--',
                         label=f'{framework}, Batch={bs}')
    
    plt.title(f'Prefill Time Comparison - {title_suffix}', fontsize=14)
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Inference Time (ms)', fontsize=12)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(sorted(df['seq_len'].unique()))  # Ensure all seq lengths are displayed
    
    plt.tight_layout()
    plt.savefig(f'{Hardware}_framework_comparison_{title_suffix.replace(" ", "_").lower()}.png', dpi=300)

# Generate plots for different batch size ranges
plot_batch_range(small_batches, "Small Batch Sizes")
plot_batch_range(medium_batches, "Medium Batch Sizes")
plot_batch_range(large_batches, "Large Batch Sizes")

# 3. Create grid of bar charts for comparing all combinations
seq_lengths = sorted(df['seq_len'].unique())
plt.figure(figsize=(20, 15))

# Calculate number of rows needed based on sequence lengths
num_rows = (len(seq_lengths) + 2) // 3  # Ceiling division to get enough rows

# For decode operation
for i, sl in enumerate(seq_lengths):
    plt.subplot(2*num_rows, 3, i+1)
    decode_data = df[(df['seq_len'] == sl) & (df['operation'] == 'decode')]
    if not decode_data.empty:
        sns.barplot(x='batch_size', y='inference_time_ms', hue='framework', data=decode_data)
        plt.title(f'Decode Time - Seq Length {sl}')
        plt.xlabel('Batch Size')
        plt.ylabel('Inference Time (ms)')
        plt.legend()
        plt.grid(axis='y')

# For prefill operation
for i, sl in enumerate(seq_lengths):
    plt.subplot(2*num_rows, 3, i+1+len(seq_lengths))
    prefill_data = df[(df['seq_len'] == sl) & (df['operation'] == 'prefill')]
    if not prefill_data.empty:
        sns.barplot(x='batch_size', y='inference_time_ms', hue='framework', data=prefill_data)
        plt.title(f'Prefill Time - Seq Length {sl}')
        plt.xlabel('Batch Size')
        plt.ylabel('Inference Time (ms)')
        plt.legend()
        plt.grid(axis='y')

plt.tight_layout()
plt.savefig(f'{Hardware}_framework_comparison_all_combinations.png', dpi=300)

# 4. Improved heatmap of speedup ratios (GQA / GLA) with proper handling of missing data
plt.figure(figsize=(20, 10))

operations = ['decode', 'prefill']
for op_idx, operation in enumerate(operations):
    plt.subplot(1, 2, op_idx+1)
    
    # Prepare data for heatmap
    heatmap_data = []
    batch_sizes = sorted(df['batch_size'].unique())
    seq_lengths = sorted(df['seq_len'].unique())
    
    for bs in batch_sizes:
        row = []
        for sl in seq_lengths:
            std_times = df[(df['framework'] == 'GQA') & 
                          (df['operation'] == operation) & 
                          (df['batch_size'] == bs) & 
                          (df['seq_len'] == sl)]['inference_time_ms'].values
            
            gla_times = df[(df['framework'] == 'GLA') & 
                          (df['operation'] == operation) & 
                          (df['batch_size'] == bs) & 
                          (df['seq_len'] == sl)]['inference_time_ms'].values
            
            if len(std_times) > 0 and len(gla_times) > 0 and gla_times[0] > 0:
                ratio = std_times[0] / gla_times[0]
                row.append(ratio)
            else:
                row.append(np.nan)  # Use NaN for missing data
                
        heatmap_data.append(row)
    
    # Create heatmap
    heatmap_df = pd.DataFrame(heatmap_data, index=batch_sizes, columns=seq_lengths)
    
    # Check if we have valid data
    if not heatmap_df.isnull().all().all():
        mask = np.isnan(heatmap_df)
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn', center=1, 
                   mask=mask, cbar_kws={'label': 'Speedup Ratio'})
        plt.title(f'{operation.capitalize()} Speedup Ratio (GQA / GLA)', fontsize=14)
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Batch Size', fontsize=12)
    else:
        plt.text(0.5, 0.5, "No valid comparison data available", horizontalalignment='center', fontsize=14)
        plt.title(f'{operation.capitalize()} Speedup Ratio (GQA / GLA)', fontsize=14)

plt.tight_layout()
plt.savefig(f'{Hardware}_framework_speedup_ratio_improved.png', dpi=300)

# 5. Create tables with raw data for reference
for operation in operations:
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 1, 1)
    plt.axis('off')
    
    # Create a table with raw data
    table_data = []
    headers = ['Batch Size', 'Seq Length']
    for framework in ['GQA', 'GLA']:
        headers.append(f'{framework} (ms)')
    headers.append('Speedup Ratio')
    
    for bs in sorted(df['batch_size'].unique()):
        for sl in sorted(df['seq_len'].unique()):
            row = [bs, sl]
            
            std_times = df[(df['framework'] == 'GQA') & 
                         (df['operation'] == operation) & 
                         (df['batch_size'] == bs) & 
                         (df['seq_len'] == sl)]['inference_time_ms'].values
            
            gla_times = df[(df['framework'] == 'GLA') & 
                         (df['operation'] == operation) & 
                         (df['batch_size'] == bs) & 
                         (df['seq_len'] == sl)]['inference_time_ms'].values
            
            if len(std_times) > 0:
                row.append(f"{std_times[0]:.2f}")
            else:
                row.append("N/A")
                
            if len(gla_times) > 0:
                row.append(f"{gla_times[0]:.2f}")
            else:
                row.append("N/A")
            
            if len(std_times) > 0 and len(gla_times) > 0 and gla_times[0] > 0:
                ratio = std_times[0] / gla_times[0]
                row.append(f"{ratio:.2f}x")
            else:
                row.append("N/A")
                
            table_data.append(row)
    
    # Create the table
    table = plt.table(cellText=table_data, 
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title(f'{operation.capitalize()} Performance Comparison Data', y=1.1, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{Hardware}_framework_{operation}_data_table.png', dpi=300)

print("Visualization complete. Check the output images.")