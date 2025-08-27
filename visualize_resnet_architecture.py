import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

def draw_conv_block(x, y, width, height, channels, label, color='lightblue'):
    rect = patches.FancyBboxPatch((x, y), width, height, 
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, f'{label}\n{channels}ch', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    return x + width

def draw_arrow(x1, y1, x2, y2, label='', style='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue', linestyle=style))
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 1, label, ha='center', fontsize=8, style='italic')

y_main = 25
block_height = 6
block_spacing = 2

ax.text(50, 45, 'Loes Scoring ResNet Architecture (MONAI Regressor)', 
        fontsize=16, fontweight='bold', ha='center')
ax.text(50, 42, '3D Brain MRI → Loes Score (0-35)', 
        fontsize=12, ha='center', style='italic')

x = 5
input_block = patches.FancyBboxPatch((x, y_main), 8, block_height,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', edgecolor='black', linewidth=1.5)
ax.add_patch(input_block)
ax.text(x + 4, y_main + block_height/2, 'Input MRI\n1×197×233×189', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(x + 4, y_main - 2, 'T1-weighted\nMNI space', 
        ha='center', va='center', fontsize=8, style='italic', color='darkgreen')
x += 8 + block_spacing

draw_arrow(x - block_spacing, y_main + block_height/2, x, y_main + block_height/2)

channels = [16, 32, 64, 128, 256, 512, 1024]
strides = [1, 2, 2, 2, 2, 1, 1]  
block_labels = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Conv7']

for i, (ch, stride, label) in enumerate(zip(channels, strides, block_labels)):
    if i < 4:
        color = 'lightblue'
    else:
        color = 'lightsalmon'
    
    x = draw_conv_block(x, y_main, 7, block_height, ch, label, color)
    
    if stride == 2:
        ax.text(x - 3.5, y_main - 2, f'stride={stride}', 
                ha='center', fontsize=7, color='red')
        ax.text(x - 3.5, y_main + block_height + 1.5, '↓2×', 
                ha='center', fontsize=8, color='red', fontweight='bold')
    
    x += block_spacing
    
    if i < len(channels) - 1:
        draw_arrow(x - block_spacing, y_main + block_height/2, x, y_main + block_height/2)
    
    if i == 3:
        res_y = y_main - 10
        res_block = patches.FancyBboxPatch((x - 9, res_y), 7, 4,
                                           boxstyle="round,pad=0.1",
                                           facecolor='lightgray', edgecolor='black', 
                                           linewidth=1, linestyle='--')
        ax.add_patch(res_block)
        ax.text(x - 5.5, res_y + 2, 'ResNet\nBlocks', 
                ha='center', va='center', fontsize=8, style='italic')
        
        draw_arrow(x - 9, y_main, x - 9, res_y + 4, style='--')
        draw_arrow(x - 2, res_y + 2, x - 2, y_main, style='--')

gap_block = patches.FancyBboxPatch((x, y_main), 7, block_height,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lavender', edgecolor='black', linewidth=1.5)
ax.add_patch(gap_block)
ax.text(x + 3.5, y_main + block_height/2, 'Global Avg\nPooling', 
        ha='center', va='center', fontsize=9, fontweight='bold')
x += 7 + block_spacing

draw_arrow(x - block_spacing, y_main + block_height/2, x, y_main + block_height/2)

fc_block = patches.FancyBboxPatch((x, y_main), 6, block_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor='gold', edgecolor='black', linewidth=1.5)
ax.add_patch(fc_block)
ax.text(x + 3, y_main + block_height/2, 'FC Layer\n1024→1', 
        ha='center', va='center', fontsize=9, fontweight='bold')
x += 6 + block_spacing

draw_arrow(x - block_spacing, y_main + block_height/2, x, y_main + block_height/2)

output_block = patches.FancyBboxPatch((x, y_main), 7, block_height,
                                      boxstyle="round,pad=0.1",
                                      facecolor='lightcoral', edgecolor='black', linewidth=1.5)
ax.add_patch(output_block)
ax.text(x + 3.5, y_main + block_height/2, 'Loes Score\n[0-35]', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(x + 3.5, y_main - 2, 'Regression\nOutput', 
        ha='center', va='center', fontsize=8, style='italic', color='darkred')

legend_y = 10
legend_items = [
    ('lightgreen', 'Input Layer'),
    ('lightblue', 'Downsampling Blocks'),
    ('lightsalmon', 'Feature Extraction'),
    ('lavender', 'Pooling'),
    ('gold', 'Dense Layer'),
    ('lightcoral', 'Output')
]

ax.text(50, legend_y + 4, 'Layer Types:', fontsize=10, fontweight='bold', ha='center')
legend_x = 25
for color, label in legend_items:
    rect = patches.Rectangle((legend_x, legend_y), 2, 1.5, 
                             facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(legend_x + 2.5, legend_y + 0.75, label, va='center', fontsize=8)
    legend_x += 12

info_y = 5
ax.text(50, info_y, 'Key Architecture Features:', fontsize=10, fontweight='bold', ha='center')
ax.text(50, info_y - 2, '• 3D Convolutions throughout • Progressive downsampling (strides=2) • 7 convolutional stages', 
        ha='center', fontsize=9)
ax.text(50, info_y - 3.5, '• Channel progression: 16→32→64→128→256→512→1024 • Global average pooling → Dense layer → Single output', 
        ha='center', fontsize=9)
ax.text(50, info_y - 5, 'Input: T1-weighted MPRAGE MRI (197×233×189 voxels) | Output: Continuous Loes score (0-35)', 
        ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('resnet_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('resnet_architecture.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Architecture diagram saved as 'resnet_architecture.png' and 'resnet_architecture.pdf'")