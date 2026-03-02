"""
train.py — One-shot training script for Layer 1 (DistilBERT on LIAR)

Run ONCE before deploying:
    python train.py

This saves the fine-tuned model to  ./models/liar_distilbert/
which is then loaded automatically by layers/text_classifier.py

Typical training time:
    CPU  → ~3-4 hours (not recommended)
    GPU  → ~8-12 minutes (T4 on Google Colab is free and sufficient)

Colab quickstart:
    !git clone https://github.com/your-handle/truthscan
    %cd truthscan
    !pip install -r requirements.txt
    !python train.py
    # Then download ./models/liar_distilbert/ and commit it to your HF Space
"""

from layers.text_classifier import train

if __name__ == "__main__":
    print("=" * 60)
    print("  TruthScan — Layer 1 Training")
    print("  DistilBERT fine-tuned on LIAR dataset (12.8k statements)")
    print("=" * 60)
    train(epochs=3, batch_size=32)
    print("\n✅ Done! Model saved to ./models/liar_distilbert/")
    print("   You can now run: python app.py")
