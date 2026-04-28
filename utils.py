import os
import matplotlib.pyplot as plt  # New
import matplotlib.patches as patches  # New

SAVE_DIR = "output_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def resize_box_xyxy(box, old_w, old_h, new_w, new_h):
    x1, y1, x2, y2 = box

    scale_x = new_w / old_w
    scale_y = new_h / old_h

    x1 *= scale_x
    y1 *= scale_y
    x2 *= scale_x
    y2 *= scale_y

    return x1, y1, x2, y2

def show_batch(images, targets, batch_idx=0):
    for i in range(len(images)):
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        boxes = targets[i]["boxes"].detach().cpu().numpy()
        labels = targets[i]["labels"].detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"class {label}",
                fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

        ax.set_title(f"Sample {i + 1} in batch")
        ax.axis("off")
        
        # ✅ Lưu ảnh ra file thay vì plt.show()
        save_path = os.path.join(SAVE_DIR, f"batch{batch_idx}_sample{i + 1}.jpg")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)  # ✅ Đóng figure, giải phóng memory
        print(f"Saved: {save_path}")
        
def plot_learning_curve(history, out_dir):                                            # NEW
    fig, ax = plt.subplots(figsize=(10, 6))                                           # NEW
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss", marker='o') # NEW
    ax.plot(history["epoch"], history["val_loss"], label="Val Loss", marker='o')     # NEW
    ax.set_xlabel("Epoch")                                                            # NEW
    ax.set_ylabel("Loss")                                                             # NEW
    ax.set_title("Learning Curve")                                                    # NEW
    ax.legend()                                                                       # NEW
    ax.grid(True)                                                                     # NEW
    plt.savefig(os.path.join(out_dir, "learning_curve.png"), bbox_inches='tight')    # NEW
    plt.close(fig)