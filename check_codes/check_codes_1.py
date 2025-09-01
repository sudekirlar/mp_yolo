import torch

# CUDA'nın kullanılabilir olup olmadığını kontrol et
is_cuda_available = torch.cuda.is_available()

print(f"PyTorch için CUDA kullanılabilir mi? -> {is_cuda_available}")

if is_cuda_available:
    # Kaç tane GPU olduğunu göster
    gpu_count = torch.cuda.device_count()
    print(f"Kullanılabilir GPU sayısı: {gpu_count}")

    # Aktif GPU'nun adını yazdır
    current_gpu_name = torch.cuda.get_device_name(0)
    print(f"Kullanılan GPU Modeli: {current_gpu_name}")
else:
    print("CUDA bulunamadı. İşlemler CPU üzerinde yapılacak.")