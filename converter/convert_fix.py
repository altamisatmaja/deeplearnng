import os
import torch
from transformers import BertForSequenceClassification, AutoConfig
from safetensors.torch import save_file

def convert_model_safely():
    try:
        print("1. Verifikasi file model...")
        model_path = "./saved_model/pytorch_model.bin"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model tidak ditemukan di {model_path}")

        print("2. Memuat config...")
        config = AutoConfig.from_pretrained("./saved_model")

        print("3. Membuat model kosong...")
        model = BertForSequenceClassification(config)

        print("4. Memuat weights (mode aman)...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("5. Memproses state_dict...")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        print("6. Memuat weights ke model...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Peringatan: {len(missing)} weights hilang")
        if unexpected:
            print(f"Peringatan: {len(unexpected)} weights tidak terduga")

        print("7. Menyimpan dalam format safetensors...")
        output_dir = "./converted_model_safe"
        os.makedirs(output_dir, exist_ok=True)
        
        save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
        
        config.save_pretrained(output_dir)
        
        print(f"\nSUKSES! Model tersimpan di {output_dir}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nSolusi alternatif:")
        print("1. Pastikan model berasal dari sumber tepercaya")
        print("2. Coba training ulang model")
        raise

if __name__ == "__main__":
    convert_model_safely()