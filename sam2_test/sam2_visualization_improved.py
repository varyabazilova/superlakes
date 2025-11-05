# Detailed comparison: Default SAM 2 vs Ground Truth (4-panel version)
if DEFAULT_TEST_DONE and GROUND_TRUTH_LOADED:
    print("📊 Creating detailed comparison of default SAM 2 vs ground truth...")

    # Combine default masks into binary array
    default_combined = np.zeros_like(ground_truth, dtype=bool)
    for mask in default_masks:
        default_combined |= mask['segmentation']

    # Calculate different mask types
    true_positive_mask = ground_truth & default_combined
    false_positive_mask = ~ground_truth & default_combined
    false_negative_mask = ground_truth & ~default_combined

    # Create 4-panel side-by-side comparison
    fig, axes = plt.subplots(1, 4, figsize=(48, 12))

    # Panel 1: Ground Truth (blue overlay only)
    axes[0].imshow(image)
    axes[0].imshow(ground_truth, alpha=0.7, cmap='Blues')
    axes[0].set_title(f'Ground Truth Lakes\n{ground_truth.sum():,} pixels ({ground_truth.mean()*100:.1f}% of image)',
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: True Positives Only (green overlay only)
    axes[1].imshow(image)
    axes[1].imshow(true_positive_mask, alpha=0.7, cmap='Greens')
    axes[1].set_title(f'Correct Detections\n{true_positive_mask.sum():,} pixels ({true_positive_mask.mean()*100:.1f}% of image)',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: False Positives Only (red overlay only)
    axes[2].imshow(image)
    axes[2].imshow(false_positive_mask, alpha=0.7, cmap='Reds')
    axes[2].set_title(f'False Positives\n{false_positive_mask.sum():,} pixels ({false_positive_mask.mean()*100:.1f}% of image)',
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Panel 4: False Negatives Only (orange overlay only)
    axes[3].imshow(image)
    axes[3].imshow(false_negative_mask, alpha=0.7, cmap='Oranges')
    axes[3].set_title(f'Missed Lakes\n{false_negative_mask.sum():,} pixels ({false_negative_mask.mean()*100:.1f}% of image)',
                     fontsize=14, fontweight='bold')
    axes[3].axis('off')

    # Add performance metrics as text
    fig.suptitle(f'SAM 2 Performance: IoU={default_metrics["iou"]:.3f}, Precision={default_metrics["precision"]:.3f}, Recall={default_metrics["recall"]:.3f}, F1={default_metrics["f1"]:.3f}',
                 fontsize=16, fontweight='bold', y=0.02)

    plt.tight_layout()
    plt.show()

    # Create detailed pixel-level analysis
    print("\n🔍 Detailed pixel-level analysis:")

    # Calculate pixel categories
    true_positive = (ground_truth & default_combined).sum()
    false_positive = (~ground_truth & default_combined).sum()
    false_negative = (ground_truth & ~default_combined).sum()
    true_negative = (~ground_truth & ~default_combined).sum()

    total_pixels = ground_truth.size

    print(f"   ✅ True Positives (correctly detected lakes): {true_positive:,} pixels")
    print(f"   ❌ False Positives (incorrectly detected as lakes): {false_positive:,} pixels")
    print(f"   ⭕ False Negatives (missed lake pixels): {false_negative:,} pixels")
    print(f"   ✅ True Negatives (correctly identified non-lakes): {true_negative:,} pixels")

    print(f"\n📈 Analysis:")
    if false_positive > false_negative:
        print(f"   🔴 SAM 2 is over-detecting (too many false positives)")
        print(f"   💡 Suggestion: Increase quality thresholds or minimum area")
    elif false_negative > false_positive:
        print(f"   🔵 SAM 2 is under-detecting (missing lakes)")
        print(f"   💡 Suggestion: Decrease thresholds or increase sampling density")
    else:
        print(f"   ⚖️ Balanced detection errors")

    # Show detailed overlap visualization (separate 4th panel)
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Create color-coded comparison
    comparison = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    comparison[ground_truth & default_combined] = [0, 255, 0]        # True positive = Green
    comparison[~ground_truth & default_combined] = [255, 0, 0]       # False positive = Red
    comparison[ground_truth & ~default_combined] = [0, 0, 255]       # False negative = Blue

    ax.imshow(image)
    ax.imshow(comparison, alpha=0.6)
    ax.set_title('Pixel-Level Error Analysis\n🟢 Correct Detection  🔴 False Positives  🔵 Missed Lakes',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\n🎯 Summary: Default SAM 2 detects {default_metrics['recall']*100:.1f}% of your lakes with {true_positive:,} correct pixels")

else:
    print("⏸️ Skipping detailed comparison (default test or ground truth not available)")